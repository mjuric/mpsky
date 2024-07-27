#!/usr/bin/env python
import os
import spiceypy as spice
import numpy as np
import astropy.units as u
from simulation_geometry import ecliptic_to_equatorial

def get_vec(row, vecname):
    """
    Extracts a vector from a Pandas dataframe row
    Parameters
    ----------
    row : row from the dataframe
    vecname : name of the vector
    Returns
    -------
    : 3D numpy array
    """
    return np.asarray([row[f"{vecname}_x"], row[f"{vecname}_y"], row[f"{vecname}_z"]])

DE440S = "de440s.bsp"
EARTH_PREDICT = "earth_200101_990825_predict.bpc"
EARTH_HISTORICAL = "earth_720101_230601.bpc"
EARTH_HIGH_PRECISION = "earth_latest_high_prec.bpc"
JPL_PLANETS = "linux_p1550p2650.440"
JPL_SMALL_BODIES = "sb441-n16.bsp"
LEAP_SECONDS = "naif0012.tls"
META_KERNEL = "meta_kernel.txt"

ORIENTATION_CONSTANTS = "pck00010.pck"
OBSERVATORY_CODES = "ObsCodes.json"

# List of kernels ordered from least to most precise - used to assemble META_KERNEL file
ORDERED_KERNEL_FILES = [
    LEAP_SECONDS,
    EARTH_HISTORICAL,
    EARTH_PREDICT,
    ORIENTATION_CONSTANTS,
    DE440S,
    EARTH_HIGH_PRECISION,
]

basepath = "/Users/mjuric/Library/Caches/sorcha"

def create_assist_ephemeris(basepath):
    """Build the ASSIST ephemeris object

    Returns
    ---------
    Ephem : ASSIST ephemeris obejct
        The ASSIST ephemeris object
    gm_sun : float
        value for the GM_SUN value
    gm_total : float
        value for gm_total
    """
    from assist import Ephem

    planets_file_path = os.path.join(basepath, JPL_PLANETS)
    small_bodies_file_path = os.path.join(basepath, JPL_SMALL_BODIES)

    ephem = Ephem(planets_path=planets_file_path, asteroids_path=small_bodies_file_path)
    gm_sun = ephem.get_particle("Sun", 0).m
    gm_total = sum(sorted([ephem.get_particle(i, 0).m for i in range(27)]))

    return ephem, gm_sun, gm_total


from orbit_conversion_utilities import universal_cartesian
import numba

def parse_orbit_row(row, epochJD_TDB, ephem, sun_dict, gm_sun, gm_total):
    """
    Parses the input orbit row, converting it to the format expected by
    the ephemeris generation code later on

    Parameters
    ---------------
    row : Pandas dataframe row
        Row of the input dataframe
    epochJD_TDB : float
        epoch of the elements, in JD TDB
    ephem: Ephem
        ASSIST ephemeris object
    sun_dict : dict
        Dictionary with the position of the Sun at each epoch
    gm_sun : float
        Standard gravitational parameter GM for the Sun
    gm_total : float
        Standard gravitational parameter GM for the Solar System barycenter

    Returns
    ------------
    : tuple
        State vector (position, velocity)

    """
    orbit_format = row["FORMAT"]

    if orbit_format not in ["CART", "BCART"]:
        if orbit_format == "COM":
            t_p_JD_TDB = row["t_p_MJD_TDB"] + 2400000.5
            ecx, ecy, ecz, dx, dy, dz = universal_cartesian(
                gm_sun,
                row["q"],
                row["e"],
                row["inc"] * np.pi / 180.0,
                row["node"] * np.pi / 180.0,
                row["argPeri"] * np.pi / 180.0,
                t_p_JD_TDB,
                epochJD_TDB,
            )
        elif orbit_format == "BCOM":
            t_p_JD_TDB = row["t_p_MJD_TDB"] + 2400000.5
            ecx, ecy, ecz, dx, dy, dz = universal_cartesian(
                gm_total,
                row["q"],
                row["e"],
                row["inc"] * np.pi / 180.0,
                row["node"] * np.pi / 180.0,
                row["argPeri"] * np.pi / 180.0,
                t_p_JD_TDB,
                epochJD_TDB,
            )
        elif orbit_format == "KEP":
            ecx, ecy, ecz, dx, dy, dz = universal_cartesian(
                gm_sun,
                row["a"] * (1 - row["e"]),
                row["e"],
                row["inc"] * np.pi / 180.0,
                row["node"] * np.pi / 180.0,
                row["argPeri"] * np.pi / 180.0,
                epochJD_TDB - (row["ma"] * np.pi / 180.0) * np.sqrt(row["a"] ** 3 / gm_sun),
                epochJD_TDB,
            )
        elif orbit_format == "BKEP":
            ecx, ecy, ecz, dx, dy, dz = universal_cartesian(
                gm_total,
                row["a"] * (1 - row["e"]),
                row["e"],
                row["inc"] * np.pi / 180.0,
                row["node"] * np.pi / 180.0,
                row["argPeri"] * np.pi / 180.0,
                epochJD_TDB - (row["ma"] * np.pi / 180.0) * np.sqrt(row["a"] ** 3 / gm_total),
                epochJD_TDB,
            )
        else:
            raise ValueError("Provided orbit format not supported.")
    else:
        ecx, ecy, ecz = row["x"], row["y"], row["z"]
        dx, dy, dz = row["xdot"], row["ydot"], row["zdot"]

    if epochJD_TDB not in sun_dict:
        sun_dict[epochJD_TDB] = ephem.get_particle("Sun", epochJD_TDB - ephem.jd_ref)

    sun = sun_dict[epochJD_TDB]

    equatorial_coords = np.array(ecliptic_to_equatorial([ecx, ecy, ecz]))
    equatorial_velocities = np.array(ecliptic_to_equatorial([dx, dy, dz]))

    if orbit_format in ["KEP", "COM", "CART"]:
        equatorial_coords += np.array((sun.x, sun.y, sun.z))
        equatorial_velocities += np.array((sun.vx, sun.vy, sun.vz))

    return tuple(np.concatenate([equatorial_coords, equatorial_velocities]))

# Note: Using JDs gives us effecive precision no worse than ~20µs
# https://aa.usno.navy.mil/downloads/novas/USNOAA-TN2011-02.pdf
def oe_to_xv(orbits_df, ephem, gm_sun, gm_total):
    # orbital elements + epoch, stored as ( x[3], v[3], epoch_jd_tdb )
    xv = np.empty((len(orbits_df), 7))

    sun_dict = dict()  # Cache for Sun position, so we avoid repeatedly calling spice
    for i, (_, row) in enumerate(orbits_df.iterrows()):
        epoch = row["epochMJD_TDB"]
        assert epoch < 2400000.5, "Inputs must be in MJD"
        epoch += 2400000.5

        xv[i, 0:6] = parse_orbit_row(row, epoch, ephem, sun_dict, gm_sun, gm_total)
        xv[i, 6] = epoch

    desig = orbits_df["ObjID"]
    return xv, desig

def create_sim(ephem, xv):
    import rebound, assist

    # Instantiate a rebound particle
    x, y, z, vx, vy, vz, epoch = xv
    ic = rebound.Particle(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)

    # Instantiate a rebound simulation and set initial time and time step
    # The time step is just a guess to start with.
    sim = rebound.Simulation()
    sim.t = epoch - ephem.jd_ref
    sim.dt = 10
    # This turns off the iterative timestep introduced in arXiv:2401.02849 and default since rebound 4.0.3
    sim.ri_ias15.adaptive_mode = 1
    # Add the particle to the simulation
    sim.add(ic)

    # Attach assist extras to the simulation
    ex = assist.Extras(sim, ephem)

    # Change the GR model for speed
    forces = ex.forces
    forces.remove("GR_EIH")
    forces.append("GR_SIMPLE")
    ex.forces = forces

    return sim, ex


def build_meta_kernel_file():
    kfn = os.path.join(basepath, META_KERNEL)

    with open(kfn, "w") as fp:
        fp.write("\\begindata\n\n")
        fp.write(f"PATH_VALUES = ('{basepath}')\n\n")
        fp.write("PATH_SYMBOLS = ('A')\n\n")
        fp.write("KERNELS_TO_LOAD=(\n")
        for fn in ORDERED_KERNEL_FILES:
            fp.write(f"    '$A/{fn}',\n")
        fp.write(")\n\n")
        fp.write("\\begintext\n")

def make_ephem_deleteme(xv_in, idx, cheby_order, tstart, tend, dt, maxabserr_meters):
    t0 = tstart
    slices = []
    while t0 < tend:
        t1 = t0 + dt
        ret, xv_in = make_ephem_slice(xv_in, idx, cheby_order, t0, t1, maxabserr_meters, return_state=True)
        slices.append( (t0, t1, ret) )
        t0 = t1

    return slices

def make_ephem_slice(xv_in, idx, cheby_order, t0, t1, maxabserr_meters, return_state=False):
#    if cheby_order > 100:
#        print(f"make_ephem :: {cheby_order=} {len(idx)=} {len(xv_in)=}")

    ntimes = cheby_order*10
    nobj = len(xv_in)
    axyz = np.empty((ntimes, 3, nobj)) # position vectors
    times = np.linspace(t0, t1, ntimes)
    mjd_ref = ephem.jd_ref - 2400000.5

    xv_out = np.empty((len(xv_in), 7))
    for oidx, xv in enumerate(xv_in):
        sim, ex = create_sim(ephem, xv)

        for tidx, t in enumerate(times):
            ex.integrate_or_interpolate(t - mjd_ref)
            target = np.array(sim.particles[0].xyz)
            axyz[tidx, :, oidx] = target

        # save the final state vector
        t = times[-1]
        ex.integrate_or_interpolate(t - mjd_ref)
        xv_out[oidx, 0:3] = np.array(sim.particles[0].xyz)
        xv_out[oidx, 3:6] = np.array(sim.particles[0].vxyz)
        xv_out[oidx, 6] = t + 2400000.5

    if cheby_order == 97:
        import pandas as pd
        df = pd.DataFrame(axyz[:, :, 0])
        df.to_csv(f"{dbg_desig.iloc[idx[0]]}.csv")

    # split into training + validation subsets
    # FIXME: optimize this, 50:50 split is likely an overkill
    axyzTest, timesTest = axyz[0::2], times[0::2]
    axyz,     times     = axyz[1::2], times[1::2]

    f = (times - t0) / (t1 - t0)
    p = np.polynomial.chebyshev.chebfit(f, axyz.reshape(axyz.shape[0], -1), cheby_order).reshape(cheby_order+1, 3, -1)

    fitok = np.ones(nobj, dtype=bool)
    if maxabserr_meters > 0: # verify that everything went well
        # axyz2.shape=(3, nobj, ntimes)
        fTest = (timesTest - t0) / (t1 - t0)
        axyz2 = np.polynomial.chebyshev.chebval(fTest, p)

        # FIXME: vectorize this
        for oidx in range(nobj):
            deltaPos = np.linalg.norm(axyz2[:, oidx, :].T - axyzTest[:, :, oidx], axis=1)
            fitok[oidx] = np.max(deltaPos) * u.au < maxabserr_meters * u.m
            if not fitok[oidx] and cheby_order > 20:
                msg = f"Approximation error too large for {dbg_desig.iloc[idx[oidx]]} @ {cheby_order=}: r_bary={np.linalg.norm(axyz2[:, oidx, 0])*u.au:7.3f}, abserr={(max(deltaPos) * u.au).to(u.cm):7.3f}"
                fitok[oidx] = True
                print(msg)

#    fitok[:] = False
    if not fitok.all():
        # return the state vectors for these
        svmask = ~fitok
        p[:, :, svmask] = np.inf

        print(f"Fit too coarse for {sum(svmask)} objects, storing state vectors instead.")
#        print("    ", p[:, :, svmask])
        xvview = p.reshape( -1, nobj)
        xvview[0, svmask] = np.inf # sentinel
        xvview[1:8, svmask] = xv_out[svmask].T
#        print("    ", p[:, :, svmask])

    ret = (idx, p)

    if return_state:
        return ret, (xv_out, idx)
    else:
        return ret

def _worker_init(basepath, desig):
    # FIXME: This is a hack. But I can't think of a better solution r.n....
    print("INITIALIZING...")
    global ephem, gm_sun, gm_total, dbg_desig
    ephem, gm_sun, gm_total = create_assist_ephemeris(basepath)
    dbg_desig = desig

def _unpack(this, **other):
    return make_ephem_slice(*this, **other)

def create_ephemerides(orbits_df, tstart, tend, dt=None, cheby_order=6, maxabserr_meters=1., map=map):
    # convert to barycentric state
    xv_in, desig_in = oe_to_xv(orbits_df, ephem, gm_sun, gm_total)

    from functools import partial
    from itertools import starmap
    from collections import defaultdict
    from tqdm import tqdm

    # split into chunks. we intentionally interleave the rows, so as to more likely
    # balance the amount of computing each thread will need to do. (e.g., nearly
    # all earth flybys were discovered relatively recently and bunch up in MPCORB)
    chunk_size = 10_000
    nchunks = int(np.ceil(len(xv_in) / chunk_size))
    idx = np.arange(len(xv_in))
    orbit_chunks = [ (xv_in[i::nchunks], idx[i::nchunks]) for i in range(nchunks) ]

    if dt is None:
        dt = tend - tstart

    t0 = tstart
    while (tend - t0) > 1./(24*3600):
        t1 = min(t0 + dt, tend)

        # integrate & fit chebys
        _func = partial(_unpack, cheby_order=cheby_order, t0=t0, t1=t1, maxabserr_meters=maxabserr_meters, return_state=True)
        chebys, indices = [], []
        orbit_chunks2 = []
        for (i, p), (xv_out, idx_out) in tqdm(map(_func, orbit_chunks), total=len(orbit_chunks)):
            orbit_chunks2.append( (xv_out, idx_out) )

            co = p.shape[0] - 1 # cheby order
            indices.append(i)
            chebys.append(p)

        # concatenate & resort
        idxout, pout = (np.concatenate(indices), np.concatenate(chebys, axis=2))
        p = np.empty_like(pout)
        p[:, :, idxout] = pout

        # store result here (TODO: yield back)
        # ...
        import pickle
        with open(f'comps.{t0}.pkl', 'wb') as fp:
            comps = (t0, t1), p, desig_in
            pickle.dump(comps, fp, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"{(t0, t1)=}")
        print(f"  total: {p.shape=}")
        svmask = p[0, 0, :] == np.inf
        print(f"  {sum(svmask)} objects with state vectors ({100*sum(svmask)/(p.shape[2]):.3}%).")

        t0 = t1
        orbit_chunks = orbit_chunks2

    return comps
#    return (t0, t1), p, desig_in

def cart_to_sph(xyz):
    x, y, z = xyz

    r = np.sqrt(x**2 + y**2 + z**2)
    ra = np.rad2deg( np.arctan2(y, x) )
    ra[ra < 0] = ra[ra < 0] + 360
    dec = np.rad2deg( np.arcsin(z/r) )

    return ra, dec

# FIXME: Speedup note: we can pre-compute and store the light-times to geocenter,
# to get a _very_ good initial guess of light-time and cut iterations down to ~1.
def get_obs(comps, t, observer, ephem, apply_ltcorr=True, lttol_msec=10, return_ephem=False, return_lt=False):
    # t is MJD in ET, either scalar or an array (one time per object)
    from simulation_constants import SPEED_OF_LIGHT
    assert not return_lt or apply_ltcorr, "get_obs: to return lt, apply_ltcorr must be set to True"

    (tmin, tmax), p, desig = comps

    nobj = len(desig)

    # extract state vectors and prepare sims
    svmask = p[0, 0, :] == np.inf
    havesv = svmask.any()

    if havesv:
        xvview = p.reshape(-1, nobj)
        xvall = xvview[1:8, svmask].T

        mjd_ref = ephem.jd_ref - 2400000.5
        sims = [ create_sim(ephem, xv) for xv in xvall ]
        nsv = len(sims)

    # observer barycentric position
    oxyz = np.atleast_2d(observer.barycentric(t)).T

    # run until light-time difference is <lttol_msec msec
    xyz = np.zeros((3, len(desig)))
    lttol = lttol_msec / 1000. / 3600. / 24.
    lt = np.zeros(nobj, dtype='f8')
    t  = t + lt # convert to ndarray
    d  = 0.
    for _ in range(5):
        # calculate light emission time
        f = (t - tmin - lt) / (tmax - tmin)
        assert np.all((0 <= f) & (f <= 1)), f"Time outside of interpolation range ({tmin=}, {tmax=}). {lt.max()=}, {np.max(d)=}."

        # position at t_emit
        with np.errstate(invalid='ignore'):
            xyz = np.polynomial.chebyshev.chebval(f, p, tensor=False)

        if havesv: # fix up objects which have state vectors, if any
            xyz2 = np.empty((3, len(sims)))
            t2 = t[svmask]
            for i, (sim, ex) in enumerate(sims):
                ex.integrate_or_interpolate(t2[i] - mjd_ref)
                xyz2[:, i] = sim.particles[0].xyz
            xyz[:, svmask] = xyz2

        # topocentric position
        xyz -= oxyz
        if not apply_ltcorr:
            # exit early if we don't need light correction
            break

        # distance at t_emit        
        d = np.linalg.norm(xyz, axis=0)

        # light flight time
        ltprev = lt
        lt = d / SPEED_OF_LIGHT

        if np.all(lt - ltprev < lttol):
            break
    else:
        assert False, "Runaway iteration in get_obs... something went wrong"

    ret = (desig, xyz)
    if return_ephem: ret += (cart_to_sph(xyz),)
    if return_lt:    ret += (lt,)
    return ret

if __name__ == "__main__":
    import sqlite3
    import pandas as pd

    # pointings
    with sqlite3.connect("baseline_v2.0_1yr.db") as con:
        pointings = pd.read_sql("select * from observations", con)

    # orbits
#    orbits = pd.read_csv("sspp_testset_orbits.des", sep=r'\s+')
    orbits = pd.read_csv("mpcorb.csv")
    print(f"{len(orbits)=}")

    # Furnish spice kernels
    build_meta_kernel_file()
    kfn = os.path.join(basepath, META_KERNEL)
    spice.furnsh(kfn)

    # Observer
    from observatory import Observatories, mjd_tai_to_et
    obs_file_path = os.path.join(basepath, OBSERVATORY_CODES)
    observer = Observatories(obs_file_path).from_obscode("X05")

    # when
#    t0 = 60507 # July 16, 2024
#    t1 = t0 + 30
#    t_tai = t0 + 0.3*(t1-t0)
##    t_tai = 60230.20135474
##    t_tai = 60225.24716783 # 2011_OB60
##    t_tai = 60230.20135474 # 2010_TU149
    t_tai = 60382.40748230 # 2012_HW1
    t = mjd_tai_to_et(t_tai)
    t0 = np.floor(t - 15)
    t1 = t0 + 30
    print(f"{(t0, t, t1)=}")
    cheby_order=9

    _worker_init(basepath, orbits["ObjID"])
    if False:
        import multiprocessing as mp
        pool = mp.Pool(processes=10, initializer=_worker_init, initargs=(basepath, orbits["ObjID"]))
        _map = map=pool.imap_unordered
    else:
        _map = map

    # compute
#    comps = create_ephemerides(orbits[:1], t0=t0, t1=t1, cheby_order=cheby_order)
    import time

    if True:
        # On an M1 Apple Silicon this takes ~0.5 msec/object, when object
        # epoch is within ~100 days of observation time
        tstart, nsamples = time.perf_counter(), 1
        for _ in range(nsamples):
            comps = create_ephemerides(orbits, tstart=t0, tend=t1, dt=30, cheby_order=cheby_order, map=_map)
        dt = time.perf_counter() - tstart
        print(f"exec_time: {dt/nsamples/len(orbits)*1000:.4} msec/object")

        import pickle
        with open('comps.pkl', 'wb') as fp:
            pickle.dump(comps, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        import pickle
        with open('comps.pkl', 'rb') as fp:
            comps = pickle.load(fp)
        print(comps[0])

#    p = comps[1]
#    svmask = p[0, 0, :] == np.inf
#    p[:, :, svmask] = 0.

#    (tmin, tmax), p, desig = comps
#    p = p[:, :, 0:1000]
#    desig = desig[0:1000]
#    comps = (tmin, tmax), p, desig

    # On an M1 Apple Silicon this takes ~0.01 msec/object with cheby_order=8
    print(t0+0.01)
    tstart, nsamples = time.perf_counter(), 20
    for _ in range(nsamples):
        desig, xyz, (ra, dec) = get_obs(comps, t + 0*(t1-0.0001), observer, ephem, return_ephem=True)
    dt = time.perf_counter() - tstart
    print(f"exec_time: {dt/nsamples/comps[1].shape[2]*1000*1000:.4} µsec/object")

    pd.set_option("display.precision", 8)
    df = pd.DataFrame({'desig': desig, 't_tai': t_tai, 'ra': ra, 'dec': dec})
    if False:
        from simulation_constants import RADIUS_EARTH_KM, AU_KM
        dfsorcha = pd.read_csv("testrun_e2e.csv")[["ObjID", "fieldMJD_TAI", "RATrue_deg", "DecTrue_deg", "Range_LTC_km"]]
        dfsorcha["Range_LTC_AU"] = dfsorcha["Range_LTC_km"] / AU_KM
        dfsorcha = dfsorcha.drop_duplicates("ObjID")

        ra0, dec0 = dfsorcha[np.abs(dfsorcha["fieldMJD_TAI"] - t_tai) < 0.00001].iloc[0][["RATrue_deg", "DecTrue_deg"]]
        df["Δra_msec"]  = (df["ra"] - ra0)*3600*1000
        df["Δdec_msec"] = (df["dec"] - dec0)*3600*1000
        df.loc[np.abs(df["Δra_msec"]) > 1,  "Δra_msec"] = np.inf
        df.loc[np.abs(df["Δdec_msec"]) > 1, "Δdec_msec"] = np.inf

        print(df)
        print(dfsorcha[:10])
    else:
        print(df)

