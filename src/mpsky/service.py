from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from logging import info, error
import time
from . import core as ac
from pydantic_settings import BaseSettings
import sys, asyncio

class Settings(BaseSettings):
    cache_path: str = "today.mpsky.bin"

settings = Settings()

from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global comps, idx
    fn = settings.cache_path
    info(f"Loading ephemerides cache from {fn}.")
    with open(fn, "rb") as fp:
        comps, idx = ac.read_comps(fp)

    info("Cache loaded.")

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
async def read_ephemerides(t: float, ra: float, dec: float, radius: float):
    # performance
    import time
    t0 = time.perf_counter()

    name, ra, dec, p, op, tmin, tmax = ac.query(comps, idx, t, ra, dec, radius)

    duration = time.perf_counter() - t0

    info(f"# objects: {len(name)}, compute time: {duration*1000:.2f}msec")

    ret = ac.ipc_write(name, ra, dec, op, p, tmin, tmax)
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
