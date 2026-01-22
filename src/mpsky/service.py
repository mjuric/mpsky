from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.responses import JSONResponse
from logging import info, error, warning, debug
import time
from . import core as ac
from pydantic_settings import BaseSettings
import sys, asyncio
import os
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Union
import aiohttp
import zstandard as zstd

class Settings(BaseSettings):
    cache_path: str = ""		# Path to local .bin file to load
    catalog_path: str = ""		# Path to local .sqlite or .csv catalog to load

    datastore_url: str = ""             # Datastore URL. Must follow the layout of gen-ephemerides-cache (https://epyc.astro.washington.edu/~mjuric/mpsky-data)
    cache_tmpdir: str = ""		# Where to store the temp files (will use a subdir in tmpdir if left empty)
    max_loaded_nights: int = 0		# How many caches (nigths) to keep in memory (default is set by mpsky serve)
    max_ondisk_nights: int = 0		# How many downloaded caches (nights) to keep on disk (default is set by mpsky serve)

settings = Settings()

# A dict of loaded caches
#    key: MJD of the night (integer)
#    value: (comps, idx, catalog) tuple
caches = OrderedDict()
def load_cache(fn, catfn):
    """ Load cache files for fn, catfn, and store them int he internal cache.
        returns (comps, idx, catalog).
    """

    info(f"Loading ephemerides cache from {fn}.")
    with open(fn, "rb") as fp:
        comps, idx = ac.read_comps(fp)

    if catfn:
        info(f"Loading the catalog from {catfn}.")
        if catfn.endswith(".sqlite"):
            import sqlite3
            con = sqlite3.connect(f"file:{catfn}?mode=ro", uri=True)

            # in tests on usdf-rsp-dev these settings seem to speed
            # up the queries by 20%-25% (~100 ms --> 75ms at best,
            # when querying on 5000 random designations)
            #
            con.execute("PRAGMA query_only=ON")
            con.executescript("""
                PRAGMA temp_store=MEMORY;
                PRAGMA mmap_size=2147483648;
                PRAGMA cache_size=-1048576;
                PRAGMA locking_mode=EXCLUSIVE;
            """)

            catalog = con
        else:
            import pandas as pd
            catalog = pd.read_csv(catfn)
            catalog.set_index("ObjID", inplace=True)
    else:
        info(f"No catalog to load")

    info("Files loaded.")

    # Now store it into the in-memory cache/index
    tmin, tmax = comps[0]
    night = ac.utc_to_night(tmin)
    assert night == ac.utc_to_night(tmax)

    caches[night] = val = ((comps, idx, catalog), (fn, catfn))
    if len(caches) > settings.max_loaded_nights:
        k, v = caches.popitem(last=False) # evict least-recent
        info(f"evicting night={k} from in-memory cache.")

        # Monitor that we truly are releasing memory...
        import weakref, gc
        w = weakref.ref(v[0][0][3]) # comps.objects
        del k, v

        if w() is not None:
            warning(f"hmm... cache evicted from dict but Python-level object not free()-d.")

    info(f"In-memory cached nights: {tuple(caches.keys())}")

    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_MB = usage.ru_maxrss / 1024
    import platform
    if platform.system() == "Darwin":
        # Darwin returns usage.ru_maxrss in bytes, Linux in kB
        rss_MB /= 1024
    info(f"Memory usage: {int(rss_MB):,}MB.")

    return val[0]

avail_caches, cache_list_expire_time = None, None
def get_datastore_cache_url(night):
    """ Find the URLs to caches in the datastore, for a given night.
    
        It caches the datastore indices so it doesn't hit it too often
        for lists of available URLs. (cache expiration: 1 minute)
    """
    import re
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    datastore_url = settings.datastore_url

    # Load all available caches
    now = datetime.now()
    global cache_list_expire_time
    global avail_caches
    if avail_caches is None or now > cache_list_expire_time :
        avail_caches = defaultdict(list)
        base_url = datastore_url + '/caches'
        pat = re.compile(rf"^eph\.[0-9]+\..*\.bin$")
        html = requests.get(base_url + "/").text
        soup = BeautifulSoup(html, "html.parser")
        cachefn = [ a["href"] for a in soup.find_all("a", href=True) if pat.match(a["href"]) ]
        if len(cachefn) == 0:
            warning(f"No caches present at {base_url}")
            return (None, None, None)
        for fn in cachefn:
            _, night_, date, _ = fn.split('.')
            night_ = int(night_)
            avail_caches[night_].append(date)
        cache_list_expire_time = now + timedelta(seconds=60)

    # Get the latest cache for the requested night
    if night in avail_caches:
        date = max(avail_caches[night])
    else:
        warning(f"no cache for {night:=} present in {datastore_url}.")
        return (None, None, None)

    cache_url = f"{datastore_url}/caches/eph.{night}.{date}.bin"
    catalog_url = f"{datastore_url}/catalogs/mpcorb-orbits.{date}.csv"
    db_url = f"{datastore_url}/catalogs/mpc_orbits.{date}.sqlite.zst"
    return (cache_url, catalog_url, db_url)

def list_to_intervals(vals):
    vals = sorted(vals)

    ranges = []
    start = prev = vals[0]

    for x in vals[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))

    out = ", ".join(
        f"{a}-{b}" if a != b else f"{a}"
        for a, b in ranges
    )

    return f"{out}"

import os
import aiohttp
import zstandard as zstd

async def _download_to(url: str, dest: str) -> None:
    """Download a URL to file using aiohttp.

    If the URL ends with '.zst', the response body is assumed to be
    zstd-compressed and is decompressed on the fly.
    """
    info(f"downloading {url} [to {os.path.basename(dest)}]...")

    dir_ = os.path.dirname(dest) or "."
    fn = os.path.basename(dest)
    tmp = os.path.join(dir_, f"tmp.{fn}")

    decompress_zstd = url.endswith(".zst")

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            r.raise_for_status()

            with open(tmp, "wb") as f:
                if not decompress_zstd:
                    async for chunk in r.content.iter_chunked(1 << 20):
                        f.write(chunk)
                else:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_writer(f, closefd=False) as zw:
                        async for chunk in r.content.iter_chunked(1 << 20):
                            zw.write(chunk)

    os.replace(tmp, dest)  # atomic rename on POSIX

cache_download_lock = asyncio.Lock()
async def get_cache(night):
    """
    Get the caches for a given night. This function is safe to call frequently.
    It will return the result from the cache, if available, and download from
    datastore otherwise.
    """
    try:
        val = caches.pop(night) # remove so we can reinsert as most-recent
        caches[night] = val
        return val[0];
    except KeyError:
        pass

    # this is for runs without a data store
    if settings.datastore_url == "":
        return next(caches.values())

    # try fetching from the datastore.
    # the lock prevents 189 clients all trying to download the same cache file.
    # FIXME: the lock really should be on a per-file basis; different files can be
    # loaded in parallel.
    async with cache_download_lock:
        # check if someone else downloaded the files while we were waiting
        # for the lock
        try:
            val = caches.pop(night) # remove so we can reinsert as most-recent
            caches[night] = val
            return val[0];
        except KeyError:
            pass

        return await _do_get_cache(night)

async def _do_get_cache(night):
    # retrieve from data store
    cache_url, catalog_url, dbfn_url = get_datastore_cache_url(night)
    if cache_url is None:
        raise Exception(f"{night=} not in available in {settings.datastore_url} (nights available: {list_to_intervals(avail_caches.keys())})")

    # compute the local cache directory for this night
    import tempfile
    if settings.cache_tmpdir != "":
        tmpdir = settings.cache_tmpdir
    else:
        import getpass
        tmpdir = tempfile.gettempdir() + f"/mpsky-caches.{getpass.getuser()}"
        os.makedirs(tmpdir, exist_ok=True)

    # set up the cache directory
    cachedir = f"{tmpdir}/night-{night}"
    try:
        os.mkdir(cachedir)
    except FileExistsError:
        os.utime(cachedir, None)  # touch for cache expiration management
    # touch a file that marks this dir as a mpsky cache dir
    CACHEDIR_SENTINEL = ".mpsky-cache-dir"
    open(f"{cachedir}/{CACHEDIR_SENTINEL}", "wb").close() 

    # construct destination filenames
    fn, catfn, dbfn = cache_url.split('/')[-1], catalog_url.split('/')[-1], dbfn_url.split('/')[-1]
    fn, catfn, dbfn = f"{cachedir}/{fn}", f"{cachedir}/{catfn}", f"{cachedir}/{dbfn}" # prefix them with 'downloaded.' so they're easy to find for eviction

    # if the upstream files are compressed, construct their decompressed names
    if not fn.endswith(".bin"):      fn = os.path.splitext(fn)[0]
    if not catfn.endswith(".csv"):   catfn = os.path.splitext(catfn)[0]
    if not dbfn.endswith(".sqlite"): dbfn = os.path.splitext(dbfn)[0]

    # fetch the files if they aren't already in cache
    # for the catalog, try fetching the .sqlite database, before
    # falling back to the old .csv
    if not os.path.exists(fn):
        await _download_to(cache_url, fn)

    if not os.path.exists(dbfn):
        try:
            await _download_to(dbfn_url, dbfn)
        except aiohttp.client_exceptions.ClientResponseError:
            info(f"couldn't download .sqlite db, falling back to .csv.")
            if not os.path.exists(catfn):
                await _download_to(catalog_url, catfn)
    if os.path.exists(dbfn):
        catfn = dbfn

    # load and cache them
    val = load_cache(fn, catfn)

    # on-disk cache cleanup. Delete old files (by modification time).
    # Note: the code above touches the mtime every time we load the cache,
    # keeping the used files safe(ish).
    import glob
    files = sorted(
        glob.glob(f"{tmpdir}/night-*"),
        key=os.path.getmtime,
        reverse=True
    )
    for dir in files[settings.max_ondisk_nights:]:
        info(f"evicting {dir} from on-disk cache.")
        import shutil
        # since we're deleting recursively (dangerous!), let's add some
        # guardrails. Only delete if we find the sentinel file.
        if os.path.exists(f"{dir}/{CACHEDIR_SENTINEL}"):
            shutil.rmtree(dir, ignore_errors=True)
        else:
            warning(f"no {CACHEDIR_SENTINEL} file in {dir}; refusing to delete it out of abundance of caution.")

    return val

async def rollover_to_new_night():
    """ Ensure the cache for current night is always loaded.
        This is only active when loading with a datastore, rather than
        directly from a single file.
    """
    current_night = 0

    while True:
        # try to load from datastore for current night. Note
        # that the current night could get evicted fromt the cache
        # if it's not used frequently; if that happens, this code
        # won't try to reload it.
        from astropy.time import Time
        night = ac.utc_to_night(Time.now().mjd)
        if night != current_night:
            info(f"rollover_to_new_night: loading current {night=}")
            await get_cache(night)
            current_night = night

        await asyncio.sleep(60)

from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    info(f"Initial cache load path: {settings.cache_path}")
    info(f"Cache data store URL: {settings.datastore_url}")
    info(f"Settings: {settings.max_loaded_nights=}, {settings.max_ondisk_nights=}")

    # preload the initial file
    if settings.cache_path != "":
        load_cache(settings.cache_path, settings.catalog_path)
    else:
        # set the periodic timer to check whether the night
        # has rolled over and load the updated cache if so.
        # note: this will also trigger immediately, loading
        # the current night.
        asyncio.create_task(rollover_to_new_night())

    yield

    info("Ephemerides server stopping.")

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    if request.url.path.startswith("/ephemerides"):
        full_request = (
            f"{request.method} {request.url.path}"
            f"{'?' + request.url.query if request.url.query else ''} "
            f"HTTP/{request.scope.get('http_version', '1.1')}"
        )
        info("Processing time {:.2f} msec [{}]".format((time.perf_counter() - start_time)*1000, full_request))
    return response

@app.exception_handler(Exception)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=400, content={"message": str(exc)})

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/version")
async def version():
    from . import _version
    return {
        "version": _version.__version__,
        "commit_id": _version.__commit_id__,
    }

from base64 import b64encode, b85encode
import pickle
import pyarrow as pa
import io

class ReturnElements(str, Enum):
    none = "none"
    basic = "basic"
    extended = "extended"

@app.get("/ephemerides/")
async def read_ephemerides(t: float, ra: float, dec: float, radius: float, return_elements: Union[bool, ReturnElements] = Query(False)):
    # performance
    import time
    t0 = time.perf_counter()

    # normalize return_elements. We take bool for backwards compatibility
    if isinstance(return_elements, bool):
        return_elements = ReturnElements.basic if return_elements else ReturnElements.none

    # chose which cache to utilize
    night = ac.utc_to_night(t)
    comps, idx, catalog = await get_cache(night)

    pass_catalog = catalog if return_elements != ReturnElements.none else None
    name, ra, dec, p, op, tmin, tmax, elements = ac.query(comps, idx, t, ra, dec, radius, pass_catalog)

    # return what's been asked for
    if return_elements == ReturnElements.basic:
        if "epoch_mjd" in elements.columns:
            elements.rename(columns={"i": "inc", "argperi": "argPeri", "peri_time":"t_p_MJD_TDB", "epoch_mjd":"epochMJD_TDB", "unpacked_primary_provisional_designation": "ObjID"}, inplace=True)
            elements = elements["q e inc node argPeri t_p_MJD_TDB epochMJD_TDB".split()]
    elif return_elements == ReturnElements.extended:
        # we allow the return to be basic, if that's all we have loaded
        pass

    duration = time.perf_counter() - t0

    info(f"# objects: {len(name)}, compute time: {duration*1000:.2f}msec")

    if elements is not None:
        info(f"# elements: {len(elements)}")

    ret = ac.ipc_write(name, ra, dec, op, p, tmin, tmax, elements)
#    ac.ipc_read(ret)
    return Response(content=ret, media_type='application/octet-stream')

    ret = pickle.dumps(
      {"name": name, "ra": ra, "dec": dec, 'ast_cheby': p, 'topo_cheby': op}
    )
    return Response(content=ret, media_type='application/octet-stream')

    ret = b64encode(
    pickle.dumps(
      {"t": name, "ra": ra, "dec": dec, 'ast_cheby': p, 'topo_cheby': op}
    )
    )
    return ret

    return {"t": name.tolist(), "ra": ra.tolist(), "dec": dec.tolist()}#, 'ast_cheby': p.tolist(), 'topo_cheby': op.tolist()}
