"""Venice web search + web scrape plugin — fleet-owned, bundled tree.

Search goes to ``POST /augment/search``; extract goes to
``POST /augment/scrape``, which returns one page as markdown. Both are sync
(httpx); the dispatcher in :mod:`tools.web_tools` handles the wrap.
"""

from __future__ import annotations

from plugins.web.venice.provider import VeniceWebSearchProvider


def register(ctx) -> None:
    """Register the Venice provider with the plugin context."""
    ctx.register_web_search_provider(VeniceWebSearchProvider())
