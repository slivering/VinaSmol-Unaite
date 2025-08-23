import asyncio
from collections import deque
import contextlib
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
import ssl
import time
from urllib.parse import urlparse
import warnings

import aiohttp
import aiohttp.client_exceptions
import certifi
import requests
from urllib3.exceptions import InsecureRequestWarning
from tqdm.asyncio import tqdm_asyncio

from .license import LicenseResult

# FIXME: this is a bit useless since requests already uses certifi.
# There are also complications in virtual environments.
# https://github.com/aio-libs/aiohttp/issues/3180#issuecomment-411739129
certifi_cafile = certifi.where()
certifi_ssl_context = ssl.create_default_context(cafile=certifi_cafile)

# https://stackoverflow.com/a/15445989
@contextlib.contextmanager
def ssl_verification(verify: bool = True):
    """Context manager to enable or disable SSL verification for requests."""
    old_merge_environment_settings = requests.Session.merge_environment_settings

    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = verify

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except Exception:
                pass

def is_ssl_certificate_valid(
        url: str,
        session: requests.Session = None,
    ) -> bool:
    try:
        if session is None:
            requests.head(url)
        else:
            match type(session):
                case requests.Session:
                    session.head(url)
                case t:
                    raise TypeError("Wrong session type", t)
        return True
    except requests.exceptions.SSLError:
        return False

async def is_ssl_certificate_valid_async(
        url: str,
        session: aiohttp.ClientSession,
    ) -> bool:
    try:
        match type(session):
            case aiohttp.ClientSession:
                await session.head(url)
            case t:
                raise TypeError("Wrong session type", t)
        return True
    except aiohttp.client_exceptions.ClientSSLError:
        return False

@contextlib.contextmanager
def auto_patch_ssl_verification(url: str, session: requests.Session):
    host = urlparse(url).hostname
    old_cert = session.cert

    secure = is_ssl_certificate_valid(url, session)
    if not secure:
        logger.info("Couldn't find SSL certificate for %s, using certifi", host)
        session.verify
        session.cert = certifi_cafile
        secure = is_ssl_certificate_valid(url, session)
        if not secure:
            logger.warning(
                "SSL certificate for %s is not recognized by certifi, "
                "requests to this host will be insecure",
                host,
            )
            session.cert = None
    
    try:
        with ssl_verification(secure):
            yield
    finally:
        session.cert = old_cert

@contextlib.asynccontextmanager
async def auto_patch_ssl_verification_async(url: str, session: aiohttp.ClientSession):
    host = urlparse(url).hostname
    old_session_connector = session._connector

    secure = await is_ssl_certificate_valid_async(url, session)
    patch_ssl_context = not secure

    if not secure:
        logger.info(
            "Couldn't find SSL certificate for %s, patching context with certifi",
            host,
        )
        session._connector = aiohttp.TCPConnector(ssl=certifi_ssl_context)
        secure = await is_ssl_certificate_valid_async(url, session)
        if not secure:
            logger.warning(
                "SSL certificate for %s is not recognized by certifi, "
                "requests to this host will be insecure",
                host,
            )
            session._connector = aiohttp.TCPConnector(ssl=False)
    
    try:
        yield
    finally:
        if patch_ssl_context:
            await session._connector.close()
            session._connector = old_session_connector


class BandwidthMeter:
    """Download bandwidth tracker with moving average."""
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.timestamp = time.monotonic()
    
    def update(self, amount: int | float) -> float:
        next_timestamp = time.monotonic()
        delta = next_timestamp - self.timestamp
        self.timestamp = next_timestamp
        self.history.append(amount / delta)

        return self.history[-1]
    
    @property
    def speed(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

@dataclass()
class DownloadResult:
    success: bool
    url: str
    file_path: Path | None = None
    content_type: str | None = None
    content_length: int = 0
    already_downloaded: bool = False


class DownloadTracker(tqdm_asyncio):
    def __init__(self, iterable=None, *args, **kwargs):
        super().__init__(iterable, *args, **kwargs)
        self._timestamp = time.monotonic()
        self._speed = 0.0
        self.__iter__()

    # https://github.com/tqdm/tqdm/issues/1286#issuecomment-2512731270
    @classmethod
    async def gather(cls, *fs, loop=None, timeout=None, total=None, return_exceptions=False, **tqdm_kwargs):
        """
        Wrapper for `asyncio.gather`.
        """
        async def wrap_awaitable(i, f):
            return i, await f

        async def wrap_awaitable_exceptions(i, f):
            try:
                return i, await f
            except Exception as e:
                return i, e

        if return_exceptions:
            ifs = [wrap_awaitable_exceptions(i, f) for i, f in enumerate(fs)]
        else:
            ifs = [wrap_awaitable(i, f) for i, f in enumerate(fs)]

        if total is None:
            total = len(fs)
        
        pbar = cls(asyncio.as_completed(ifs, timeout=timeout), total=total, **tqdm_kwargs)
        results = []

        meter = BandwidthMeter(window_size=20)
        for f in pbar:
            i, result = await f
            results.append((i, result))
            if isinstance(result, DownloadResult):
                url = "..." + "/".join(result.url.split("/")[-2:])
                if result.success:
                    meter.update(result.content_length / 2**20)
                pbar.set_postfix(url=url, speed=f"{meter.speed:.2g} MB/s", refresh=True)
            elif isinstance(result, LicenseResult):
                url = "..." + "/".join(result.url.split("/")[-2:])
                pbar.set_postfix(url=url, refresh=True)
            elif isinstance(result, Exception):
                url = "Download error"
                meter.update(0.0)
                pbar.set_postfix(url=url, refresh=True)

        return [res for _, res in sorted(results)]

