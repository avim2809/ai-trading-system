"""Investing.com Pro authenticated-scraper data source.

Investing.com has no official API, so this package obtains its data via an
authenticated ``requests.Session`` (see :mod:`firm.data.investing.session`)
against the site's own pages/endpoints rather than a documented API.

Off by default: every fetcher in this package raises
:class:`firm.data.providers.base.ProviderError` unless
``INVESTING_SCRAPER_ENABLED`` is set truthy — with it unset, nothing in this
package touches the network. See ``docs/investing_pro_integration.md`` for
the risk/benefit rationale and the enablement checklist.
"""

from __future__ import annotations
