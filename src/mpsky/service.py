from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from logging import info, error, warning
import time
from . import core as ac
from pydantic_settings import BaseSettings
import sys, asyncio
import os
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta

class Settings(BaseSettings):
    cache_path: str = ""
    catalog_path: str = ""
    cache_datastore: str = "https://epyc.astro.washington.edu/~mjuric/mpsky-data"	# datastore URL. Must follow the layout of gen-ephemerides-cache
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
        info(f"Evicting night={k} from in-memory cache.")
    info(f"In-memory cached nights: {tuple(caches.keys())}")
    
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

    cache_datastore = settings.cache_datastore

    # Load all available caches
    now = datetime.now()
    global cache_list_expire_time
    global avail_caches
    if avail_caches is None or now > cache_list_expire_time :
        avail_caches = defaultdict(list)
        base_url = cache_datastore + '/caches'
        pat = re.compile(rf"^eph\.[0-9]+\..*\.bin$")
        html = requests.get(base_url).text
        soup = BeautifulSoup(html, "html.parser")
        cachefn = [ a["href"] for a in soup.find_all("a", href=True) if pat.match(a["href"]) ]
        if len(cachefn) == 0:
            warning(f"No caches present at {base_url}")
            return (None, None)
        for fn in cachefn:
            _, night_, date, _ = fn.split('.')
            night_ = int(night_)
            avail_caches[night_].append(date)
        cache_list_expire_time = now + timedelta(seconds=60)

    # Get the latest cache for the requested night
    try:
        date = max(avail_caches[night])
    except KeyError:
        warning(f"No cache for {night:=} present at {base_url}")
        return (None, None)

    cache_url = f"{cache_datastore}/caches/eph.{night}.{date}.bin"
    catalog_url = f"{cache_datastore}/catalogs/mpcorb-orbits.{date}.csv"
    return (cache_url, catalog_url)

def get_cache(night):
    """
    Get the files for a given night. This function is safe to call frequently.
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
    if settings.cache_datastore == "":
        return next(caches.values())

    # retrieve from data store
    cache_url, catalog_url = get_datastore_cache_url(night)
    if cache_url is None:
        raise Exception(f"({night:=}) not in available night ranges {tuple(avail_caches.keys())}")

    # compute the local cache directory
    import tempfile
    if settings.cache_tmpdir != "":
        tmpdir = settings.cache_tmpdir
    else:
        import getpass
        tmpdir = tempfile.gettempdir() + f"/mpsky-caches.{getpass.getuser()}"
        os.makedirs(tmpdir, exist_ok=True)

    fn, catfn = cache_url.split('/')[-1], catalog_url.split('/')[-1]
    fn, catfn = f"{tmpdir}/downloaded.{fn}", f"{tmpdir}/downloaded.{catfn}" # prefix them with 'downloaded.' so they're easy to find for eviction

    # fetch the files, caching them locally; skip if they're already fetched
    # NOTE: this download is intentionally not async, for now. That ensures that
    # we don't get 189 threads all trying to download the same cache at once.
    # There are (of course) ways to work around this, but I don't have time to do
    # it r.n. and we don't really need it for the use case.
    import urllib.request
    if not os.path.exists(fn):
        info(f"{fn} <- {cache_url} ...")
        tmpfn, _ = urllib.request.urlretrieve(cache_url)
        os.rename(tmpfn, fn)
        info("done")
    else:
        os.utime(fn, None) # touch the file modification time, for cache management
        info(f"{fn} already downloaded.")

    if not os.path.exists(catfn):
        info(f"{catfn} <- {catalog_url}")
        tmpcatfn, _ = urllib.request.urlretrieve(catalog_url)
        os.rename(tmpcatfn, catfn)
        info(f"done.")
    else:
        os.utime(catfn, None) # touch the file modification time, for cache management
        info(f"{catfn} already downloaded.")

    # load and cache them
    val = load_cache(fn, catfn)

    # on-disk cache cleanup. Delete old files (by modification time).
    # Note: the code above touches the mtime every time we load the cache,
    # keeping the used files safe(ish).
    import glob
    files = sorted(
        glob.glob(f"{tmpdir}/downloaded.*"),
        key=os.path.getmtime,
        reverse=True
    )
    MAX_FILE_CACHE = 2 * settings.max_ondisk_nights # Number of files per night, times number of nights to allow on disk
    for fn in files[MAX_FILE_CACHE:]:
        info(f"Deleting {fn} from on-disk cache.")

    return val


async def rollover_to_new_night():
    """ Ensure the cache for current night is always loaded.
        This is only active when loading with a datastore, rather than
        directly from a single file.
    """
    while True:
        # try to load from datastore for current night
        from astropy.time import Time
        night = ac.utc_to_night(Time.now().mjd)
        if night not in caches:
            info(f"rollover_to_new_night: loading current {night=}")
            get_cache(night)
        else:
            info(f"rollover_to_new_night: {night=} already loaded.")

        await asyncio.sleep(10)

from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    info(f"Initial cache load path: {settings.cache_path}")
    info(f"Cache data store URL: {settings.cache_datastore}")
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
    if request.url.path != "/":
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

from base64 import b64encode, b85encode
import pickle
import pyarrow as pa
import io

@app.get("/ephemerides/")
async def read_ephemerides(t: float, ra: float, dec: float, radius: float, return_elements: bool = False):
    # performance
    import time
    t0 = time.perf_counter()

    # chose which cache to utilize
    night = ac.utc_to_night(t)
    comps, idx, catalog = get_cache(night)

    pass_catalog = catalog if return_elements else None
    name, ra, dec, p, op, tmin, tmax, elements = ac.query(comps, idx, t, ra, dec, radius, pass_catalog)

    duration = time.perf_counter() - t0

    info(f"# objects: {len(name)}, compute time: {duration*1000:.2f}msec")

    if elements is not None:
        info(f"# elements: {len(elements)}")

    ret = ac.ipc_write(name, ra, dec, op, p, tmin, tmax, elements)
    ac.ipc_read(ret)
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
