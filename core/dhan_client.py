"""
Async HTTP wrapper for the Dhan trading API (v2).

Uses httpx.AsyncClient with rate limiting, structured error handling,
and typed return values for all market-data endpoints.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta

import httpx

from core.timezone import now_ist
from config import DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID, DHAN_API_BASE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class DhanAPIError(Exception):
    """Raised when a Dhan API call fails."""

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        self.status_code = status_code
        self.body = body
        super().__init__(message)


# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

NIFTY_SECURITY_ID = 13
INDIA_VIX_SECURITY_ID = 21  # Was 26 (wrong) — confirmed 21 from Dhan scrip master

# Minimum gap (seconds) between consecutive API calls.
_RATE_LIMIT_SECONDS = 0.10


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class DhanClient:
    """Async wrapper around the Dhan v2 REST API."""

    def __init__(self) -> None:
        self._base = DHAN_API_BASE.rstrip("/")
        self._headers = {
            "access-token": DHAN_ACCESS_TOKEN,
            "Content-Type": "application/json",
            "client-id": DHAN_CLIENT_ID,
        }
        self._client: httpx.AsyncClient | None = None
        self._last_request_ts: float = 0.0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Create the underlying httpx session."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base,
                headers=self._headers,
                timeout=httpx.Timeout(30.0),
            )
            logger.info("DhanClient session opened (%s)", self._base)

    async def close(self) -> None:
        """Gracefully close the httpx session."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("DhanClient session closed")

    async def __aenter__(self) -> "DhanClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # -- rate limiting -------------------------------------------------------

    async def _throttle(self) -> None:
        """Ensure at least _RATE_LIMIT_SECONDS between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if elapsed < _RATE_LIMIT_SECONDS:
            await asyncio.sleep(_RATE_LIMIT_SECONDS - elapsed)
        self._last_request_ts = time.monotonic()

    # -- low-level request ---------------------------------------------------

    async def _post(self, path: str, payload: dict, retries: int = 2) -> dict:
        """
        POST *path* with *payload* and return the parsed JSON response.

        Retries on 429 (rate limit) with exponential backoff.
        Raises DhanAPIError on HTTP or network errors.
        """
        if self._client is None or self._client.is_closed:
            await self.start()

        for attempt in range(retries + 1):
            await self._throttle()

            try:
                resp = await self._client.post(path, json=payload)
            except httpx.RequestError as exc:
                logger.error("Network error on POST %s: %s", path, exc)
                raise DhanAPIError(f"Network error: {exc}") from exc

            if resp.status_code == 401:
                body = resp.text
                # Distinguish subscription errors from auth errors
                if "806" in body or "DH-902" in body or "not subscribed" in body.lower():
                    logger.error(
                        "Data API subscription inactive (401) — "
                        "subscribe at web.dhan.co → Profile → Access DhanHQ APIs"
                    )
                    raise DhanAPIError(
                        "Data APIs not subscribed — activate at web.dhan.co",
                        status_code=401,
                        body=body,
                    )
                logger.error("Authentication failed (401) — token expired or invalid")
                raise DhanAPIError(
                    "Authentication failed — generate new token at web.dhan.co",
                    status_code=401,
                    body=body,
                )

            if resp.status_code == 429 and attempt < retries:
                wait = 2 ** (attempt + 1)  # 2s, 4s
                logger.warning(f"Rate limited (429) on {path}, retrying in {wait}s (attempt {attempt + 1}/{retries})")
                await asyncio.sleep(wait)
                continue

            if resp.status_code >= 400:
                logger.error(
                    "Dhan API error %s on POST %s: %s",
                    resp.status_code,
                    path,
                    resp.text[:500],
                )
                raise DhanAPIError(
                    f"HTTP {resp.status_code} on {path}",
                    status_code=resp.status_code,
                    body=resp.text,
                )

            try:
                return resp.json()
            except ValueError as exc:
                raise DhanAPIError("Invalid JSON in response") from exc

        # Should not reach here, but just in case
        raise DhanAPIError(f"All {retries + 1} attempts failed for {path}")

    # -- token renewal -------------------------------------------------------

    async def renew_token(self) -> dict:
        """
        Attempt to renew the Dhan access token for another 24 hours.

        NOTE: As of Feb 2026, Dhan's RenewToken endpoint appears to not work
        for API portal-generated tokens (DH-905/DH-906 errors). The scheduled
        renewal job will log this failure gracefully. The pipeline still works
        without token renewal — it falls back to parquet data.

        If Dhan fixes this in the future, this method is ready to go.
        """
        await self._throttle()

        url = f"{self._base}/RenewToken"
        headers = {
            "access-token": self._headers["access-token"],
            "dhanClientId": DHAN_CLIENT_ID,
        }

        logger.info("Attempting token renewal at %s", url)

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                resp = await client.post(url, headers=headers)
                logger.info("RenewToken response: %s %s", resp.status_code, resp.text[:300])
        except httpx.RequestError as exc:
            logger.error("Network error on RenewToken: %s", exc)
            raise DhanAPIError(f"Network error during token renewal: {exc}") from exc

        if resp.status_code >= 400:
            body = resp.text[:500]
            logger.error("Token renewal failed (HTTP %s): %s", resp.status_code, body)
            raise DhanAPIError(
                f"Token renewal failed (HTTP {resp.status_code}): {body}",
                status_code=resp.status_code,
                body=resp.text,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise DhanAPIError("Invalid JSON in RenewToken response") from exc

        new_token = data.get("accessToken")
        if not new_token:
            raise DhanAPIError(f"No accessToken in renewal response: {data}")

        # Update in-memory headers for this client and the httpx session
        self._headers["access-token"] = new_token
        if self._client and not self._client.is_closed:
            self._client.headers["access-token"] = new_token

        expiry = data.get("expiryTime", "unknown")
        logger.info("Dhan token renewed successfully — expires %s", expiry)

        return data

    # -- public endpoints ----------------------------------------------------

    async def get_nifty_ltp(self) -> float | None:
        """Return the last-traded price for Nifty 50 index, or None on error."""
        try:
            prices = await self._get_index_prices()
            return prices.get("nifty")
        except DhanAPIError:
            logger.exception("Failed to fetch Nifty LTP")
            return None

    async def get_india_vix(self) -> float | None:
        """Return the last-traded price for India VIX, or None on error."""
        try:
            prices = await self._get_index_prices()
            return prices.get("vix")
        except DhanAPIError:
            logger.exception("Failed to fetch India VIX")
            return None

    async def _get_index_prices(self) -> dict:
        """
        Fetch Nifty 50 and India VIX in a SINGLE API call.

        Returns dict with keys 'nifty' and 'vix' (float values).
        Caches for 5 seconds to avoid duplicate calls when get_nifty_ltp()
        and get_india_vix() are called back-to-back.
        """
        now = time.monotonic()

        # Return cached result if fresh (< 5 seconds old)
        if (hasattr(self, "_index_cache") and
                self._index_cache and
                now - self._index_cache_ts < 5.0):
            return self._index_cache

        data = await self._post(
            "/marketfeed/ltp",
            {"IDX_I": [NIFTY_SECURITY_ID, INDIA_VIX_SECURITY_ID]},
        )

        # Response shape: {"data": {"IDX_I": {"13": {"last_price": 25535}, "21": {"last_price": 12.4}}}}
        prices = {}
        try:
            idx_data = data.get("data", {}).get("IDX_I", {})
            nifty_data = idx_data.get(str(NIFTY_SECURITY_ID), {})
            vix_data = idx_data.get(str(INDIA_VIX_SECURITY_ID), {})

            if "last_price" in nifty_data:
                prices["nifty"] = float(nifty_data["last_price"])
            if "last_price" in vix_data:
                prices["vix"] = float(vix_data["last_price"])
        except (TypeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse index prices from {data}: {e}")

        if prices:
            logger.info(f"Index prices: Nifty={prices.get('nifty')}, VIX={prices.get('vix')}")
        else:
            logger.warning(f"Empty index prices from response: {data}")

        self._index_cache = prices
        self._index_cache_ts = now
        return prices

    async def get_option_chain(self, expiry_date: str) -> dict | None:
        """
        Fetch the Nifty option chain for *expiry_date* (``YYYY-MM-DD``).

        Returns the raw JSON dict, or None on error.
        """
        try:
            return await self._post(
                "/optionchain",
                {
                    "UnderlyingScrip": NIFTY_SECURITY_ID,
                    "UnderlyingSeg": "IDX_I",
                    "Expiry": expiry_date,
                },
            )
        except DhanAPIError:
            logger.exception("Failed to fetch option chain for %s", expiry_date)
            return None

    async def get_historical_ohlc(self, days: int = 365) -> list | None:
        """
        Fetch daily OHLC candles for the Nifty 50 index going back *days* days.

        Returns a list of candle dicts, or None on error.
        """
        to_date = now_ist().strftime("%Y-%m-%d")
        from_date = (now_ist() - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            data = await self._post(
                "/charts/historical",
                {
                    "securityId": NIFTY_SECURITY_ID,
                    "exchangeSegment": "IDX_I",
                    "instrument": "INDEX",
                    "expiryCode": 0,
                    "fromDate": from_date,
                    "toDate": to_date,
                },
            )
            return self._extract_candles(data)
        except DhanAPIError:
            logger.exception("Failed to fetch historical OHLC")
            return None

    async def get_intraday_ohlc(self, interval: str = "5") -> list | None:
        """
        Fetch intraday OHLC candles for the Nifty 50 index.

        *interval*: candle width in minutes (``"1"``, ``"5"``, ``"15"``, ``"25"``, ``"60"``).

        Returns a list of candle dicts, or None on error.
        """
        try:
            data = await self._post(
                "/charts/intraday",
                {
                    "securityId": NIFTY_SECURITY_ID,
                    "exchangeSegment": "IDX_I",
                    "instrument": "INDEX",
                    "interval": interval,
                    "fromDate": now_ist().strftime("%Y-%m-%d"),
                    "toDate": now_ist().strftime("%Y-%m-%d"),
                },
            )
            return self._extract_candles(data)
        except DhanAPIError:
            logger.exception("Failed to fetch intraday OHLC")
            return None

    async def get_option_ltp(self, security_id: str) -> float | None:
        """Return the LTP for a single FnO instrument by *security_id*, or None."""
        try:
            data = await self._post(
                "/marketfeed/ltp",
                {"NSE_FNO": [int(security_id)]},
            )
            return self._extract_ltp(data)
        except DhanAPIError:
            logger.exception("Failed to fetch option LTP for %s", security_id)
            return None

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _extract_ltp(data: dict) -> float:
        """
        Pull the LTP float out of a ``/marketfeed/ltp`` response.

        The Dhan response nests the price inside ``data -> <segment> -> <secId>``.
        We walk the dict to find the first numeric ``last_price`` / ``ltp`` value.
        """
        # Response shape: {"data": {"IDX_I": {"13": {"last_price": 22345.6}}}}
        # or similar nested structure — walk until we find a float-like value.
        if isinstance(data, dict):
            for key in ("last_price", "ltp", "LTP", "lastPrice"):
                if key in data:
                    return float(data[key])
            for value in data.values():
                if isinstance(value, dict):
                    try:
                        return DhanClient._extract_ltp(value)
                    except (ValueError, TypeError, KeyError):
                        continue
                elif isinstance(value, (int, float)):
                    return float(value)

        raise ValueError(f"Could not extract LTP from response: {data}")

    @staticmethod
    def _extract_candles(data: dict) -> list:
        """
        Normalise a ``/charts/*`` response into a list of candle dicts.

        Dhan returns parallel arrays: open[], high[], low[], close[], volume[], timestamp[].
        We zip them into a list of dicts.
        """
        if not isinstance(data, dict):
            return []

        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        closes = data.get("close", [])
        volumes = data.get("volume", [])
        timestamps = data.get("timestamp", data.get("start_Time", []))

        if not opens:
            return []

        candles = []
        for i in range(len(opens)):
            candles.append(
                {
                    "open": opens[i],
                    "high": highs[i] if i < len(highs) else None,
                    "low": lows[i] if i < len(lows) else None,
                    "close": closes[i] if i < len(closes) else None,
                    "volume": volumes[i] if i < len(volumes) else None,
                    "timestamp": timestamps[i] if i < len(timestamps) else None,
                }
            )
        return candles
