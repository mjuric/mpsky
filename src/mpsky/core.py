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

def ipc_write(name, ra, dec, op, p):
    # fast pyarrow IPC serialization
    outbuf = io.BytesIO()
    out = pa.output_stream(outbuf)
    a = pa.Tensor.from_numpy(p);   pa.ipc.write_tensor(a, out)
    a = pa.Tensor.from_numpy(op);  pa.ipc.write_tensor(a, out)
    data = [ pa.array(name), pa.array(ra), pa.array(dec) ]
    batch = pa.record_batch(data, names=['name', 'ra', 'dec'])
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

    return r["name"].to_numpy(zero_copy_only=False), r["ra"].to_numpy(), r["dec"].to_numpy(), p.to_numpy(), op.to_numpy()

def utc_to_night(mjd, obscode='X03'):
    assert obscode == 'X03'
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

def compress(df, cheby_order = 4, observer_cheby_order = 7):
    # make sure the input is sorted by ObjID and time.
    df = df.sort_values(["ObjID", "FieldMJD_TAI"])
    objects = df["ObjID"].unique()
    nobj = len(objects)
    nobs = len(df) // len(objects)
    assert len(df) % nobj == 0, "All objects must have been observed at the same times"

    # extract times
    t = df["FieldMJD_TAI"].values[0:nobs]
    tmin, tmax = t.min(), t.max()
    t -= tmin
    assert np.all(np.round(t) == 0), "Hmmm... the adjusted times should span [0, 1) day range"

    #
    # extract & compress the topocentric observer vector
    #
    oxyz = np.empty((nobs, 3))
    oxyz[:, 0] = (df["Obs-Sun(J2000x)(km)"].values * u.km).to(u.au).value[0:nobs]
    oxyz[:, 1] = (df["Obs-Sun(J2000y)(km)"].values * u.km).to(u.au).value[0:nobs]
    oxyz[:, 2] = (df["Obs-Sun(J2000z)(km)"].values * u.km).to(u.au).value[0:nobs]
    op = np.polynomial.chebyshev.chebfit(t, oxyz, observer_cheby_order)

    # Check that the decompressed topocentric position makes sense
    oxyz2 = np.polynomial.chebyshev.chebval(t, op)
    assert np.all(np.abs(oxyz2/oxyz.T - 1) < 5e-7)

    #
    # Fit asteroid chebys
    #
    axyz = np.empty((nobs, 3, nobj))
    axyz[:, 0, :].T.flat = (df["Ast-Sun(J2000x)(km)"].values * u.km).to(u.au).value
    axyz[:, 1, :].T.flat = (df["Ast-Sun(J2000y)(km)"].values * u.km).to(u.au).value
    axyz[:, 2, :].T.flat = (df["Ast-Sun(J2000z)(km)"].values * u.km).to(u.au).value
    p = np.polynomial.chebyshev.chebfit(t, axyz.reshape(nobs, -1), cheby_order).reshape(cheby_order+1, 3, -1)

    # Check that the decompressed asteroid positions make sense
    axyz2 = np.polynomial.chebyshev.chebval(t, p)
    ra  = df['AstRA(deg)'].values
    dec = df['AstDec(deg)'].values

    xyz = axyz2 - oxyz2[:, np.newaxis, :] # --> (xyz, objid, nobs)
    x, y, z = xyz

    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.rad2deg( np.arcsin(z/r) ).flatten()
    lon = np.rad2deg( np.arctan2(y, x) ).flatten()

    dd = haversine(lon, lat, ra, dec)*3600
    assert dd.max() < 1

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

def write_comps(fp, comps, idx):
    pickle.dump(comps, fp, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(idx, fp, protocol=pickle.HIGHEST_PROTOCOL)

def read_comps(fp):
    comps, idx = pickle.load(fp), pickle.load(fp)
    idx = JaggedArray.from_dict(idx)
    return comps, idx

def _aux_compress(fn, nside=128, verify=True, tolerance_arcsec=1):
    df = pd.read_hdf(fn)

    # Extract a dataframe only for the specific night,
    # or (if night hasn't been given) verify the input only has a single night
    nights = utc_to_night(df["FieldMJD_TAI"].values)
    assert np.all(nights == nights[0]), "All inputs must come from the same night"

    comps = compress(df)
    idx = build_healpix_index(comps, nside)

    if verify:
        # extract visit times for this night
        df2 = df.sort_values(["ObjID", "FieldMJD_TAI"])
        t = df2["FieldMJD_TAI"].values[ df2["ObjID"] == df2["ObjID"].iloc[0] ]
        ra  = df2['AstRA(deg)'].values
        dec = df2['AstDec(deg)'].values
        objects, _, (ra2, dec2) = decompress(t, comps, return_ephem=True)
        ra2, dec2 = ra2.flatten(), dec2.flatten()
        dd = haversine(ra2, dec2, ra, dec)*3600
        assert dd.max() < tolerance_arcsec

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

def query(comps, idx, t, ra, dec, radius, use_index=True):
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
    return name, ra, dec, p, op

def query_service(url, t, ra, dec, radius):
    params = {
        "t": t,
        "ra": ra,
        "dec": dec,
        "radius": radius
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
            name, ra, dec, p, op = query_service(args.source, args.t, args.ra, args.dec, args.radius)
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
        name, ra, dec, p, op = query(comps, idx, args.t, args.ra, args.dec, args.radius)
        duration = time.perf_counter() - t0

    if args.format == "json":
        import json
        js = json.dumps({'name': name.tolist(), 'ra:': ra.tolist(), 'dec': dec.tolist(), 'ast_cheby': p.tolist(), 'topo_cheby': op.tolist()})
        print(js)
    elif args.format == "table":
        # print the results
        dist = haversine(ra, dec, args.ra, args.dec)
        print("#   object            ra           dec          dist")
        for n, r, d, dd in zip(name, ra, dec, dist):
            print(f"{n:10s} {r:13.8f} {d:13.8f} {dd:13.8f}")
        assert np.all(dist <= args.radius)
        print(f"# objects: {len(name)}")
        print(f"# compute time: {duration*1000:.2f}msec")
        print(f"# source: {args.source}")
    else:
        assert False, f"uh, oh, this should not happen. Format {args.format=} is unrecognized."

def main():
    import argparse

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
    parser_serve.add_argument('cache_path', type=str, nargs='?', default="today.mpsky.pkl", help='Cache file to read from')
    parser_serve.add_argument('--host', type=str, default="127.0.0.1", help='Hostname or IP to bind to.')
    parser_serve.add_argument('--port', type=int, default=8000, help='Port to bind to.')
    parser_serve.add_argument('--reload', action='store_true', default=False, help='Automatically reload.')
    parser_serve.add_argument('--log-config', type=str, help='Uvicorn logging configuration file.')
    parser_serve.add_argument('--verbose', action='store_true', default=False, help='Activate verbose logging.')

    # Create the parser for the "query" command
    parser_query = subparsers.add_parser('query', help='Query data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_query.add_argument('t', type=float, help='Time (MJD, UTC)')
    parser_query.add_argument('ra', type=float, help='Right ascension (degrees)')
    parser_query.add_argument('dec', type=float, help='Declination (degrees)')
    parser_query.add_argument('--radius', type=float, default=1, help='Search radius (degrees)')
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
