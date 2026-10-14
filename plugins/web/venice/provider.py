"""Venice web search + page-to-markdown scrape — fleet-owned provider plugin.

Subclasses :class:`plugins.web._common.BaseWebSearchProvider`. Two capabilities:

- ``supports_search()``  -> True  (``POST https://api.venice.ai/api/v1/augment/search``)
- ``supports_extract()`` -> True  (``POST https://api.venice.ai/api/v1/augment/scrape``)

Both routes are Venice's own "augment" API. Search answers ``{query,
results: [{title, url, content, date}]}``; scrape answers one page as
``{url, content, format: "markdown"}`` and takes exactly one URL per request,
so :meth:`extract` fans out one request per URL and keeps the request order.

Config keys this provider responds to::

    web:
      search_backend: "venice"    # or web.backend: "venice" for both
      extract_backend: "venice"

Env vars::

    VENICE_MCP_KEY=...            # required; the key the veniceMCP server already uses

No fallback and no keyless tier: a missing key or a non-2xx answer surfaces the
vendor's own text (matching ``web.keyless_fallback: false``). Page content and
URLs are never written to the log — only counts.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import httpx

from plugins.web._common import (
    SEARCH_LIMIT_CAP,
    BaseWebSearchProvider,
    document,
    page_error,
    provider_env,
    run_extract,
    run_search,
    search_ok,
    setup_schema,
    web_hit,
)

logger = logging.getLogger(__name__)

_KEY_ENV = "VENICE_MCP_KEY"
_BASE_URL = "https://api.venice.ai/api/v1"
_SEARCH_URL = f"{_BASE_URL}/augment/search"
_SCRAPE_URL = f"{_BASE_URL}/augment/scrape"

_SEARCH_TIMEOUT = 30
_SCRAPE_TIMEOUT = 60

# The scrape route returns no title; a leading markdown H1 becomes the title when present.
_NO_TITLE = ""
_MAX_TITLE_CHARS = 200


def _api_key() -> str:
    key = provider_env(_KEY_ENV)
    if not key:
        raise ValueError(f"{_KEY_ENV} is not set")
    return key


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _error_text(response: httpx.Response) -> str:
    """Vendor error text: the body's ``error``/``message`` field when the body is JSON, else the raw body."""
    body = (response.text or "").strip()
    if body.startswith("{"):
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        if isinstance(data, dict):
            for key in ("error", "message", "detail"):
                value = data.get(key)
                if isinstance(value, str) and value:
                    return value
    return body or f"HTTP {response.status_code}"


def _post(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """POST one augment request; raise ``ValueError`` carrying the vendor's own error text on non-2xx."""
    response = httpx.post(url, json=payload, headers=_headers(_api_key()), timeout=timeout)
    if response.status_code >= 400:
        raise ValueError(_error_text(response))
    return response.json()


def _title_from_markdown(content: str) -> str:
    """First ``# heading`` line of the scraped markdown, else ``""``."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()[:_MAX_TITLE_CHARS]
        if stripped and not stripped.startswith(("#", "---", "[")):
            break
    return _NO_TITLE


class VeniceWebSearchProvider(BaseWebSearchProvider):
    """Venice augment API: web search rows plus full-page markdown extract."""

    NAME = "venice"
    DISPLAY_NAME = "Venice AI"
    KEY_ENV = _KEY_ENV
    EXTRACT = True
    KEYLESS = False

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Return up to *limit* web rows from ``POST /augment/search``."""

        def body() -> dict[str, Any]:
            wanted = max(1, min(int(limit), SEARCH_LIMIT_CAP))
            data = _post(_SEARCH_URL, {"query": query, "limit": wanted}, _SEARCH_TIMEOUT)
            raw_results = data.get("results") or []
            rows = [
                web_hit(
                    str(result.get("url") or ""),
                    str(result.get("title") or ""),
                    str(result.get("content") or ""),
                    index + 1,
                )
                for index, result in enumerate(raw_results[:wanted])
            ]
            logger.info("Venice search: %d result(s) returned", len(rows))
            return search_ok(rows)

        return run_search(self.DISPLAY_NAME, logger, body)

    def extract(self, urls: list[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Scrape each URL to markdown, one request per URL, in request order.

        A URL Venice refuses (``400`` for sites that block automated access) or
        answers without content becomes an entry carrying ``error``; a missing
        key fails every URL. Unknown ``kwargs`` are ignored.
        """

        def body() -> list[dict[str, Any]]:
            api_key = _api_key()
            documents: list[dict[str, Any]] = []
            failures = 0
            for url in urls:
                response = httpx.post(
                    _SCRAPE_URL, json={"url": url}, headers=_headers(api_key), timeout=_SCRAPE_TIMEOUT
                )
                if response.status_code >= 400:
                    documents.append(page_error(url, _error_text(response)))
                    failures += 1
                    continue
                data = response.json()
                content = str(data.get("content") or "")
                if not content:
                    documents.append(page_error(url, "no content returned"))
                    failures += 1
                    continue
                documents.append(document(url, _title_from_markdown(content), content))
            if failures:
                logger.warning("Venice scrape: %d of %d URL(s) failed", failures, len(urls))
            logger.info("Venice scrape: %d/%d URL(s) returned content", len(urls) - failures, len(urls))
            return documents

        return run_extract(self.DISPLAY_NAME, logger, urls, body)

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "Venice AI",
            "paid",
            "Venice augment API — web search rows plus a page-to-markdown scrape. ~$0.01 per request.",
            _KEY_ENV,
            "Venice API key",
            "https://venice.ai/settings/api",
            web_tier="paid",
        )
