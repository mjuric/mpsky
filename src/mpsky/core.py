#!/usr/bin/env python

import pandas as pd
import astropy.units as u
import numpy as np
import healpy as hp
import pickle
import io
import pyarrow as pa
import requests
import time
import sys
import os
import struct

# because astropy is slow AF
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return np.degrees(c)

ELEMENTS_LIST   = "q e inc node argPeri t_p_MJD_TDB epochMJD_TDB".split()
ELEMENTS_FORMAT = " ".join([ "{:> 11f}" ] * len(ELEMENTS_LIST))
HEADER_FORMAT = "{:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>13s}  {:>12s}"

def ipc_write(name, ra, dec, op, p, tmin, tmax, elements):
    # fast pyarrow IPC serialization
    outbuf = io.BytesIO()
    out = pa.output_stream(outbuf)
    a = pa.Tensor.from_numpy(p);   pa.ipc.write_tensor(a, out)
    a = pa.Tensor.from_numpy(op);  pa.ipc.write_tensor(a, out)
    data = [ pa.array(name), pa.array(ra), pa.array(dec), pa.array(np.ones_like(ra) * tmin), pa.array(np.ones_like(ra) * tmax)]
    names = ['name', 'ra', 'dec', 'tmin', 'tmax']

    # add the columns from elements, if passed
    if elements is not None:
        for col in ELEMENTS_LIST:
            data.append(pa.array(elements[col].values))
            names.append(col)

    batch = pa.record_batch(data, names=names)
    with pa.ipc.new_stream(out, batch.schema) as writer:
      writer.write_batch(batch)
    return outbuf.getvalue()

def ipc_read(msg):
    with pa.input_stream(memoryview(msg)) as fp:
        fp.seek(0)
        p  = pa.ipc.read_tensor(fp)
        op = pa.ipc.read_tensor(fp)
        with pa.ipc.open_stream(fp) as reader:
            schema = reader.schema
            r = next(reader)

    # construct an elements pandas dataframe
    if "q" in schema.names:
        cols = { col: r[col].to_numpy() for col in ELEMENTS_LIST }
        elements = pd.DataFrame(cols)
    else:
        elements = None

    return r["name"].to_numpy(zero_copy_only=False), r["ra"].to_numpy(), r["dec"].to_numpy(), p.to_numpy(), op.to_numpy(), elements

def utc_to_night(mjd, obscode='X05'):
    assert obscode == 'X05'
    localtime = mjd - 4./24.  ## hack to convert UTC to ~approx local time for Chile (need to do this better...)
    night = (localtime - 0.5).astype(int)
    return night

def build_healpix_index(comps, nside, dt_minutes=5):
    #
    # Computes a dictionary where the key are healpix indices
    # (NSIDE, nested) and the values are the lists (ndarrays)
    # of objects (their indices, actually) that have passed
    # through that pixel in the period covered by the interpolation.
    # This is done by computing the position of the object from
    # tmin to tmax, evedy dt_minutes minutes.
    #
    # The returned dictionary lets the user quickly get a
    # list of asteroids that passed through a given pixel.
    #
    # Example:
    #   > h2l = build_healpix_index(comps, nside=128)
    #   > print(h2l[5000])
    #
    #   [   1739   20004  223389  418207  824376  880008 1062034 1252353]
    #

    # compute position vector
    (tmin, tmax), op, p, objects = comps
    t = np.arange(tmin, tmax, dt_minutes/(24*60))
    objects, xyz = decompress(t, comps, return_ephem=False)

    # compute healpix pixel corresponding to this vector
    x, y, z = xyz
    ipix = hp.vec2pix(nside, x, y, z, nest=True)

    # object IDs corresponding to each ipix entry
    #     shape = (len(objects), len(t))
    # it looks like:
    #    array([[      0,       0,       0, ...,       0,       0,       0],
    #           [      1,       1,       1, ...,       1,       1,       1],
    #            ...,
    i = np.tile(np.arange(ipix.shape[0]), (ipix.shape[1], 1)).T

    # flatten
    ipix = ipix.reshape(-1)
    i = i.reshape(-1)
    
    # Now the goal is to jointly sort the ipix and i array, so that ipix is the key and i is the value
    idx = np.argsort(ipix)
    ipix_sorted = ipix[idx]
    astid_sorted = i[idx]

    # Now we build the { hpix -> [ ast_id ] } dictionary -- this is our index.
    # initialize dict mapping to empty arrays
    h2l = dict( map(lambda key: (key, np.zeros(0, dtype=int)), range(hp.nside2npix(nside))) )

    # fill out pixels where there are asteroids
    hpix, limits, counts = np.unique(ipix_sorted, return_index=True, return_counts=True)
    for h, l, c in zip(hpix, limits, counts):
        h2l[h] = np.unique(astid_sorted[l:l+c])

    return h2l

def compress(df, cheby_order = 7, observer_cheby_order = 7):
    # make sure the input is sorted by ObjID and time.
    df = df.sort_values(["ObjID", "fieldMJD_TAI"])
    objects = df["ObjID"].unique()
    nobj = len(objects)
    nobs = len(df) // len(objects)
    assert len(df) % nobj == 0, "All objects must have been observed at the same times"

    # extract times
    t = df["fieldMJD_TAI"].values[0:nobs]
    tmin, tmax = t.min(), t.max()
    t -= tmin
    assert np.max(t) <= 1.0#np.all(np.round(t) == 0), "Hmmm... the adjusted times should span [0, 1) day range"

    #
    # extract & compress the topocentric observer vector
    #
    oxyz = np.empty((nobs, 3))
    oxyz[:, 0] = (df["Obs_Sun_x_km"].values * u.km).to(u.au).value[0:nobs]
    oxyz[:, 1] = (df["Obs_Sun_y_km"].values * u.km).to(u.au).value[0:nobs]
    oxyz[:, 2] = (df["Obs_Sun_z_km"].values * u.km).to(u.au).value[0:nobs]
    op = np.polynomial.chebyshev.chebfit(t, oxyz, observer_cheby_order)

    # Check that the decompressed topocentric position makes sense
    oxyz2 = np.polynomial.chebyshev.chebval(t, op)
    assert np.all(np.abs(oxyz2/oxyz.T - 1) < 5e-7)

    #
    # Fit asteroid chebys
    #
    axyz = np.empty((nobs, 3, nobj))
    axyz[:, 0, :].T.flat = (df["Obj_Sun_x_LTC_km"].values * u.km).to(u.au).value
    axyz[:, 1, :].T.flat = (df["Obj_Sun_y_LTC_km"].values * u.km).to(u.au).value
    axyz[:, 2, :].T.flat = (df["Obj_Sun_z_LTC_km"].values * u.km).to(u.au).value
    p = np.polynomial.chebyshev.chebfit(t, axyz.reshape(nobs, -1), cheby_order).reshape(cheby_order+1, 3, -1)

    # Check that the decompressed asteroid positions make sense
    axyz2 = np.polynomial.chebyshev.chebval(t, p)
    ra  = df['RA_deg'].values
    dec = df['Dec_deg'].values

    xyz = axyz2 - oxyz2[:, np.newaxis, :] # --> (xyz, objid, nobs)
    x, y, z = xyz

    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.rad2deg( np.arcsin(z/r) ).flatten()
    lon = np.rad2deg( np.arctan2(y, x) ).flatten()

    dd = haversine(lon, lat, ra, dec)*3600
    print('error 50, 90, 99, 99.9, 100:', np.percentile(dd, (50, 90, 99, 99.9, 100)))
    #assert dd.max() < 2

    #
    # return the results
    #
    return (tmin, tmax), op, p, objects

def cart_to_sph(xyz):
    x, y, z = xyz

    r = np.sqrt(x**2 + y**2 + z**2)
    ra = np.rad2deg( np.arctan2(y, x) )
    ra[ra < 0] = ra[ra < 0] + 360
    dec = np.rad2deg( np.arcsin(z/r) )

    return ra, dec

def decompress(t_mjd, comps, return_ephem=False):
    (tmin, tmax), op, p, objects = comps

    # adjust the time, and assert we're within the range of interpolation validity
    if not np.all((tmin <= t_mjd) & (t_mjd <= tmax)):
        raise Exception(f"The interpolation is valid from {tmin} to {tmax}")
    t = t_mjd - tmin

    oxyz2 = np.polynomial.chebyshev.chebval(t, op)  # Decompress topo position
    axyz2 = np.polynomial.chebyshev.chebval(t, p)   # Decompress asteroid position
    xyz = axyz2 - oxyz2[:, np.newaxis]              # Obs-Ast vector

    if not return_ephem:
        return objects, xyz
    else:
        return objects, xyz, cart_to_sph(xyz)

def merge_comps(compslist):
    from tqdm import tqdm

    # verify tmin/tmax are the same everywhere
    for i, (comps, idx) in enumerate(compslist):
        assert comps[0] == compslist[0][0][0], f"Interpolation limits don't match, {comps[0]} != {compslist[0][0][0]} at index={i}"
        assert np.all(comps[1] == compslist[0][0][1]), f"Observer location chebys don't match, {comps[1]} != {compslist[0][0][1]} at index={i}"
    (tmin, tmax), op, _, _ = comps

    p = [ comps[2] for comps, _ in compslist]
    p = np.concatenate(p, axis=2)

    # convert to a string ndarray
    from itertools import chain
    objects = [ comps[3] for comps, _ in compslist ]
    objects = list(chain(*objects))
    objects = np.asarray(objects)

    # merge indices
    allidx = dict( map(lambda key: (key, []), range(len(idx))) )
    delta = 0
    for comps, idx in tqdm(compslist):
        _, _, _, o = comps
        for i in range(len(allidx)):
            allidx[i].append(idx[i] + delta)
        delta += len(o)
    for i in range(len(allidx)):
        allidx[i] = np.concatenate(allidx[i])
    idx = allidx
    idx = JaggedArray.from_dict(idx)
    #print('keys:', idx.keys())
    #print(len(idx))

    comps = (tmin, tmax), op, p, objects
    return comps, idx

class JaggedArray:
    def __init__(self, vals, offs):
        self.vals = vals
        self.offs = offs
        self.n = len(self.offs) - 1

    def __getitem__(self, i):
        return self.vals[self.offs[i]:self.offs[i+1]]

    def __len__(self):
        return self.n

    @staticmethod
    def from_dict(idx):
        # construct offset array
        nvals = sum(map(len, idx.values()))
        vals = np.empty(nvals, int)
        offs = np.empty(len(idx)+1, int)
        a = 0
        for i, v in enumerate(idx.values()):
            offs[i] = a
            b = a + len(v)
            vals[a:b] = v
            a = b
        offs[-1] = a
        return JaggedArray(vals, offs)

ALIGN_BYTES = 64

def write_numpy(fp, a):
    # allow lists of arrays: concatenate them efficiently
    # on write-out
    at = 0

    concat = isinstance(a, list) or isinstance(a, tuple)
    if not concat: a = [ a ]

    # write dtype
    b = a[0].dtype.str.encode("ascii")
    b = struct.pack('<q', len(b)) + b
    at += fp.write(b)

    # write shape (prefixed by shape length)
    shape = a[0].shape if not concat else (len(a),) + a[0].shape
    at += fp.write(np.array((len(shape),) + shape).data)

    # pad to alignment
    at += fp.write(b'0'*(ALIGN_BYTES - (at % ALIGN_BYTES)))
    assert at % ALIGN_BYTES == 0, at

    # write array contents
    for arr in a:
        at += fp.write(arr.data)

    # pad to alignment
    at += fp.write(b'0'*(ALIGN_BYTES - (at % ALIGN_BYTES)))
    assert at % ALIGN_BYTES == 0, at
    
    return at

def read_numpy(fp, mode='r', mmap=True):
    # read dtype
    l = struct.unpack('<q', fp.read(8))[0]
    dtype = np.dtype(fp.read(l).decode("ascii"))
    
    # read shape
    l = struct.unpack('<q', fp.read(8))[0]
    shape = tuple(np.frombuffer(fp.read(l*8), dtype=int))

    # consume padding
    l = ALIGN_BYTES - (fp.tell() % ALIGN_BYTES)
    fp.read(l)

    if mmap:
        # memorymap the data
        offset = fp.tell()
        arr = np.memmap(fp, dtype=dtype, mode=mode, shape=shape, offset=offset)

        # seek beyond the array data + padding
        at = offset + arr.nbytes
        fp.seek(at + ALIGN_BYTES - (at % ALIGN_BYTES))
    else:
        nbytes = np.prod(shape) * dtype.itemsize
        arr = np.frombuffer(fp.read(nbytes), dtype=dtype).reshape(shape)

        at = fp.tell()
        fp.seek(at + ALIGN_BYTES - (at % ALIGN_BYTES))

    return arr

def fake_multinight(comps, idx, n=5):
    # While hacking multi-night support...
    (tmin, tmax), op, p, objects = comps
    tminmax = np.array([ (tmin + i, tmax + i) for i in range(n) ])
    op      = np.tile(op, (n, 1, 1))
    p       = np.tile(p, (n, 1, 1, 1))
    objects = np.tile(objects, (n, 1))

    idx = [ JaggedArray(idx.vals, idx.offs) for _ in range(n) ]

    return (tminmax, op, p, objects), idx

#def write_comps(fp, comps, idx):
#    pickle.dump(comps, fp, protocol=pickle.HIGHEST_PROTOCOL)
#    pickle.dump(idx, fp, protocol=pickle.HIGHEST_PROTOCOL)
#
#def read_comps(fp):
#    comps, idx = pickle.load(fp), pickle.load(fp)
#    idx = JaggedArray.from_dict(idx)
##    comps, idx = fake_multinight(comps, idx)
##    with open('today.mpsky.bin', 'wb') as fp:
##        write_comps_ng(fp, comps, idx)
#    return comps, idx

def write_comps(fp, comps, idx):
    tminmax, op, p, objects = comps
    at = 0
    at += write_numpy(fp, np.asarray(tminmax))
    at += write_numpy(fp, op)
    at += write_numpy(fp, p)
    at += write_numpy(fp, objects)

    if isinstance(idx, dict):
        idx = JaggedArray.from_dict(idx)

    if isinstance(idx, JaggedArray):
        vals, offs = idx.vals, idx.offs
    else:
        vals = [ i.vals for i in idx ] #idx[i]
        offs = [ i.offs for i in idx ]
    at += write_numpy(fp, vals)
    at += write_numpy(fp, offs)

    return at

def read_comps(fp, mmap=True):
    tminmax = read_numpy(fp)
    op = read_numpy(fp)
    p = read_numpy(fp, mmap=mmap)
    objects = read_numpy(fp, mmap=mmap)
    comps = tminmax, op, p, objects

    vals = read_numpy(fp, mmap=mmap)
    offs = read_numpy(fp, mmap=mmap)
    if vals.ndim == 1:
        idx = JaggedArray(vals, offs)
    else:
        idx = [ JaggedArray(v, o) for v, o in zip(vals, offs) ]
    return comps, idx

def _aux_compress(fn, nside=128, verify=True, tolerance_arcsec=1):
    df = pd.read_hdf(fn)

    # Extract a dataframe only for the specific night,
    # or (if night hasn't been given) verify the input only has a single night
    nights = utc_to_night(df["fieldMJD_TAI"].values)
    # assert np.all(nights == nights[0]), "All inputs must come from the same night"

    comps = compress(df)
    idx = build_healpix_index(comps, nside)

    if verify:
        # extract visit times for this night
        df2 = df.sort_values(["ObjID", "fieldMJD_TAI"])
        t = df2["fieldMJD_TAI"].values[ df2["ObjID"] == df2["ObjID"].iloc[0] ]
        ra  = df2['RA_deg'].values
        dec = df2['Dec_deg'].values
        objects, _, (ra2, dec2) = decompress(t, comps, return_ephem=True)
        ra2, dec2 = ra2.flatten(), dec2.flatten()
        dd = haversine(ra2, dec2, ra, dec)*3600
        print('max error', dd.max())
        #assert dd.max() < tolerance_arcsec

    return comps, idx

def fit_many(fns, ncores):
    from tqdm import tqdm
    from functools import partial
    from multiprocessing import Pool
    with Pool(processes=ncores) as pool:
        all_comps_and_idx = list(tqdm(pool.imap(_aux_compress, fns), total=len(fns)))

    return merge_comps(all_comps_and_idx)

def cmd_build(args):
    import time

    outfn = args.output # f'cache.mjd={night0}.pkl'
    fns = args.ephem_file # '/astro/store/epyc3/data3/jake_dp03/for_mario/mpcorb_eph_*.hdf')
    ncores = args.j

    comps, idx = fit_many(fns, ncores=ncores)

    with open(outfn, "wb") as fp:
        write_comps(fp, comps, idx)
    import os
    print(f"wrote {outfn} [ size={os.stat(outfn).st_size:,}]")

    # decompress for a single time
    print("single decompress (no ephem)...", end='')
    t0 = time.time()
    decompress((comps[0][0] + comps[0][1])/2, comps, return_ephem=False)
    duration = time.time() - t0
    print(f" done [{duration:.2f}sec]")

    print("Success!")

def find_comp(comps, idx, t):
    if not isinstance(idx, list):
        tmin, tmax = tminmax = comps[0]
        if tmin <= t and t <= tmax:
            return comps, idx
    else:
        tminmax, op, p, objects = comps
        i = np.searchsorted(tminmax[:, 0], t)
        if 0 < i and i < len(tminmax):
            i -= 1
            tmin, tmax = tminmax[i]
            if t <= tmax:
                return (tminmax[i], op[i], p[i], objects[i]), idx[i]

    raise Exception(f"t={t} not in available ranges ({tminmax})")


def query(comps, idx, t, ra, dec, radius, catalog):
    # find the right night
    comps, idx = find_comp(comps, idx, t)

    # compute pointing vector
    radius = np.radians(radius)
    ra_rad, dec_rad = np.radians(ra), np.radians(dec)
    pointing = np.asarray([ np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad) ])

    if idx is not None:
        # find plausible asteroids
        nside = hp.npix2nside(len(idx))
        hpix = hp.query_disc(nside, pointing, radius=radius, inclusive=True, nest=True)
        ast = np.unique(np.concatenate([ idx[k] for k in hpix ]))

        # extract chebys only for plausible asteroids
        (tmin, tmax), op, p, objects = comps
        comps2 = ((tmin, tmax), op, p[:, :, ast], objects[ast])
    else:
        comps2 = comps

    # decompress for a single time
    objects, xyz = decompress(t, comps2, return_ephem=False)

    # turn to a unit vector
    r = np.sqrt((xyz*xyz).sum(axis=0))
    xyz /= r

    # query the position via dot-product
    cos_radius = np.cos(radius)
    dotprod = (xyz.T*pointing).sum(axis=1)
    mask = dotprod > cos_radius

    # select the results
    _, op, p, _ = comps2
    name, (ra, dec), p = objects[mask], cart_to_sph(xyz[:, mask]), p[:, :, mask]
    
    # match elements, if requested
    if catalog is not None:
        elements = catalog.loc[name]
    else:
        elements = None

    return name, ra, dec, p, op, tmin, tmax, elements

def query_service(url, t, ra, dec, radius, return_elements):
    params = {
        "t": t,
        "ra": ra,
        "dec": dec,
        "radius": radius,
        "return_elements": return_elements
    }

    # Sending a GET request to the endpoint
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        print("failed to connect to the remote ephemerides service. details:", file=sys.stderr)
        print(e, file=sys.stderr)
        exit(-1)

    return ipc_read(response.content)

def cmd_serve(args):
    # This will be read by the Settings in the service
    import os
    os.environ["CACHE_PATH"] = args.cache_path
    os.environ["CATALOG_PATH"] = args.catalog

    if args.verbose:
        import os.path
        assert args.log_config is None, "--verbose and --log-config cannot be specified at the same time"
        args.log_config = os.path.join(os.path.dirname(__file__), "log_conf.yaml")

    import uvicorn
    config = uvicorn.Config("mpsky.service:app", host=args.host, port=args.port, log_level="info", log_config=args.log_config, reload=args.reload)
    server = uvicorn.Server(config)
    server.run()

def cmd_query(args):
    if args.source.startswith("http://") or args.source.startswith("https://"):
        # remote service query
        assert not args.no_index, "Only valid for local queries"
        try:
            t0 = time.perf_counter()
            name, ra, dec, p, op, elements = query_service(args.source, args.t, args.ra, args.dec, args.radius, args.return_elements)
            duration = time.perf_counter() - t0
        except requests.exceptions.HTTPError as e:
            print(f"Error (status={e.response.status_code}): {e.response.text}", file=sys.stderr)
            return -1
    else:
        # local file query
        with open(args.source, "rb") as fp:
            comps, idx = read_comps(fp)
        if args.no_index:
            idx = None

        t0 = time.perf_counter()
        name, ra, dec, p, op, elements = query(comps, idx, args.t, args.ra, args.dec, args.radius, catalog)
        duration = time.perf_counter() - t0

    if args.format == "json":
        import json
#        js = json.dumps({'name': name.tolist(), 'ra:': ra.tolist(), 'dec': dec.tolist(), 'ast_cheby': p.tolist(), 'topo_cheby': op.tolist()})
#        print(js)
        cols = dict(name=name, ra=ra, dec=dec)
        df = pd.DataFrame(cols)
        if elements is not None:
            df = pd.concat([df, elements], axis=1)
        df["ast_cheby"] = [ p[:, :, i].T.tolist() for i in range(p.shape[2]) ]
        data = {
            "ast": df.to_dict(orient="records"),
            "topo_cheby": op.T.tolist()
        }
#        data = dict(ast=df, topo_cheby=op.T.tolist())
#        print(df.to_json(orient="records"))
        print(json.dumps(data))
    elif args.format == "table":
        # print the results
        dist = haversine(ra, dec, args.ra, args.dec)
        if elements is None:
            print("#   object            ra           dec          dist")
            for n, r, d, dd in zip(name, ra, dec, dist):
                print(f"{n:10s} {r:13.8f} {d:13.8f} {dd:13.8f}")
        else:
            print(("#   object            ra           dec          dist " + HEADER_FORMAT).format(*elements.columns))
            for values in zip(name, ra, dec, dist, *elements.to_numpy().T):
                print(("{:10s} {:13.8f} {:13.8f} {:13.8f} " + ELEMENTS_FORMAT).format(*values))
        assert np.all(dist <= args.radius)
        print(f"# objects: {len(name)}")
        print(f"# query time: {duration*1000:.2f}msec")
        print(f"# source: {args.source}")
    else:
        assert False, f"uh, oh, this should not happen. Format {args.format=} is unrecognized."

def main():
    import argparse
    import signal

    # don't vomit exceptions when a pipe is broken (i.e., when piped to `head`)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    # Create the top-level parser
    parser = argparse.ArgumentParser(description='Asteroid Checker.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommands')

    # Create the parser for the "compress" command
    parser_build = subparsers.add_parser('build', help='Compress ephemerides files.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_build.add_argument('ephem_file', type=str, nargs='+', help='T')
    parser_build.add_argument('-j', type=int, default=1, help='Run multithreaded')
    parser_build.add_argument('--output', type=str, required=True, help='Output file name.')

    # Create the parser for the "serve" command
    # Shorthand for running `uvicorn service:app --reload --log-config=log_conf.yaml`
    parser_serve = subparsers.add_parser('serve', help='Serve data via an HTTP interface', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_serve.add_argument('cache_path', type=str, nargs='?', default="today.mpsky.bin", help='Cache file to read from')
    parser_serve.add_argument('--host', type=str, default="127.0.0.1", help='Hostname or IP to bind to.')
    parser_serve.add_argument('--port', type=int, default=8000, help='Port to bind to.')
    parser_serve.add_argument('--reload', action='store_true', default=False, help='Automatically reload.')
    parser_serve.add_argument('--log-config', type=str, help='Uvicorn logging configuration file.')
    parser_serve.add_argument('--verbose', action='store_true', default=False, help='Activate verbose logging.')
    parser_serve.add_argument('--catalog', type=str, default="", help='Catalog file with additional data corresponding to the objects in cache.')

    # Create the parser for the "query" command
    parser_query = subparsers.add_parser('query', help='Query data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_query.add_argument('t', type=float, help='Time (MJD, UTC)')
    parser_query.add_argument('ra', type=float, help='Right ascension (degrees)')
    parser_query.add_argument('dec', type=float, help='Declination (degrees)')
    parser_query.add_argument('--radius', type=float, default=1, help='Search radius (degrees)')
    parser_query.add_argument('--return-elements', action='store_true', help='Return the orbital elements with the ephemerides')
    parser_query.add_argument('--no-index', action='store_true', default=False, help='Do not use the healpix index.')
    parser_query.add_argument('--format', type=str, choices=['table', 'json'], default='table', help='Output format.')
    url = os.getenv("MPSKY_URL", 'https://sky.dirac.dev/ephemerides/')
    parser_query.add_argument('--source', type=str, nargs='?', const=url, default=url, help=f'Local ephemerides cache file or service endpoint URL.')

    # Parse the arguments
    args = parser.parse_args()

    # Check which command is being requested and call the appropriate function/handler
    if args.command == 'build':
        return cmd_build(args)
    elif args.command == 'query':
        return cmd_query(args)
    elif args.command == 'serve':
        return cmd_serve(args)

if __name__ == '__main__':
    main()
