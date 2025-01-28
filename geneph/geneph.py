#!/usr/bin/env python

import os
import spiceypy as spice
import numpy as np

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

basepath = "/Users/jake/Library/Caches/sorcha"

JPL_PLANETS = "linux_p1550p2650.440"
JPL_SMALL_BODIES = "sb441-n16.bsp"

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

def build_assist(basepath):
    """Build the ASSIST ephemeris object

    Returns
    ---------
    Ephem : ASSIST ephemeris obejct
        The ASSIST ephemeris object
    gm_sun : float
        value for the GM_SUN value (used to convert heliocentric elements)
    gm_total : float
        value for gm_total (used to convert barycentric elements)
    """
    from assist import Ephem

    planets_file_path = os.path.join(basepath, JPL_PLANETS)
    small_bodies_file_path = os.path.join(basepath, JPL_SMALL_BODIES)

    ephem = Ephem(planets_path=planets_file_path, asteroids_path=small_bodies_file_path)
    gm_sun = ephem.get_particle("Sun", 0).m
    # FIXME: 27 shouldn't be hardcoded, but there isn't a call in ASSIST to get this number
    gm_total = sum(sorted([ephem.get_particle(i, 0).m for i in range(27)]))

    return ephem, gm_sun, gm_total

def cart_to_sph(xyz):
    # convert cartesian xyz to spherical (lon, lat)

    x, y, z = xyz.T

    r = np.sqrt(x**2 + y**2 + z**2)
    ra = np.rad2deg( np.arctan2(y, x) )
    ra[ra < 0] = ra[ra < 0] + 360
    dec = np.rad2deg( np.arcsin(z/r) )

    return ra, dec

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
    sim.ri_ias15.min_dt = 1e-10
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

def parse_orbit_row(row, epochJD_TDB, ephem, sun_dict, gm_sun, gm_total):
    """
    Parses the input orbit row, converting it to the format expected by
    the ephemeris generation code later on

    Parameters
    ---------------
    row : dict or numpy array row
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
    from simulation_geometry import ecliptic_to_equatorial
    from orbit_conversion_utilities import universal_cartesian

    orbit_format = row["FORMAT"]
    if isinstance(orbit_format, bytes):
        orbit_format = str(orbit_format, encoding='ascii')

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
def oe_to_xv(orbits, ephem, gm_sun, gm_total):
    # orbital elements + epoch, stored as ( x[3], v[3], epoch_jd_tdb )
    xv = np.empty((len(orbits), 7))

    sun_dict = dict()  # Cache for Sun position, so we avoid repeatedly calling spice
    for i, row in enumerate(orbits):
        epoch = row["epochMJD_TDB"]
        assert epoch < 2400000.5, "Inputs must be in MJD"
        epoch += 2400000.5

        xv[i, 0:6] = parse_orbit_row(row, epoch, ephem, sun_dict, gm_sun, gm_total)
        xv[i, 6] = epoch

    desig = orbits["ObjID"]
    return xv, desig

def propagate_states(xv_in, times, ephem):
    # given barycentric state vectors in xv_in[nobj, 7]
    # propagate them to all MJDs in the vector times.
    #
    # output: xv[nobj, times, 7]
    #
    # state vector format: [x, y, z, vx, vy, vz, epochJD_TDB]

    nobj = len(xv_in)
    ntimes = len(times)
    xv = np.empty((nobj, ntimes, 7)) # output state vectors
    mjd_ref = ephem.jd_ref - 2400000.5

    # for each object
    for oidx, oxv in enumerate(xv_in):
        sim, ex = create_sim(ephem, oxv)

        # for each time
        for tidx, t in enumerate(times):
            ex.integrate_or_interpolate(t - mjd_ref)
            xv[oidx, tidx, 0:3] = np.array(sim.particles[0].xyz)
            xv[oidx, tidx, 3:6] = np.array(sim.particles[0].vxyz)
            xv[oidx, tidx, 6] = t + 2400000.5

    return xv

def eval_cheby(pobj, t, per_object_times=False):
    # input (tmin, tmax, p) -- the chebishev coefficients
    # p.shape = (nobj, 7, cheby+1)
    #
    # output: xv[nobj, ntimes, 7] or xv[nobj, 7] (if per_object_times=True)
    #

    # unpack pobj
    (tmin, tmax, p) = pobj

    # rescale to [-1, 1]
    f = (2.*t - (tmin + tmax)) / (tmax - tmin)

    assert np.all((-1. <= f) & (f <= 1.)), f"Time outside of interpolation range ({tmin=}, {tmax=}). {t.min()=}, {t.max()=}."

    p_reshaped = p.transpose(2, 0, 1)
    if not per_object_times:
        xv = np.polynomial.chebyshev.chebval(f, p_reshaped).transpose(0, 2, 1)
        print("xv_eval =======")
        print(xv.shape) # (nobj, ntimes, 7)
        print(xv[0,  0])
        print(xv[0,  1])
        print(xv[0, -1])
    else:
        p_reshaped = p.transpose(2, 1, 0)
        xv = np.polynomial.chebyshev.chebval(f, p_reshaped, tensor=False).T

    return xv

def fit_cheby(t, xv, cheby_order, tmin=None, tmax=None):
    # input: - an array of position or state vectors in the format xv[nobj, times, 7]
    # or xv[nobj, times, 3]
    # - t: times of sample points
    # - order of the chebyshev polynomial to fit
    #
    # output: 
    #
    #   (tmin, tmax, coefs)
    #   where coefs.shape = (...)
    #   and the polynomial is fit to [-1, 1] range (rescaled using tmin, tmax)
    #
    nobj = len(xv)
    ntimes = len(t)

    tmin = t.min() if tmin is None else tmin
    tmax = t.max() if tmax is None else tmax

    # rescale to [-1, 1]
    f = (2.*t - (tmin + tmax)) / (tmax - tmin)
    print("fffffffffffffffffffff")
    print(f.min(), f.max())

    # fit polynomials
    # We need to reshape to nobj, times, 7 --> times, nobj, 7 and then flatten,
    # as chebfit expects to find the dat as columns of the input dataset.
    xv_reshaped = np.transpose(xv, (1, 0, 2)).reshape(ntimes, -1) 
    p = np.polynomial.chebyshev.chebfit(f, xv_reshaped, cheby_order).reshape(cheby_order+1, nobj, -1)

    # transpose to (nobj, 7, cheby+1) shape, to keep all the data belonging to the same object
    # close by in memory (and on disk). This will be our canonical storage format.
    p = np.transpose(p, (1, 2, 0))
    print("p =======")
    print(p.shape)
    print(p[0])

    return (tmin, tmax, p)

def estimate_accuracy(pobj, times, xv):
    #
    # compute the differences between computed and decompressed
    # and evaluate the expected maximum astrometry error when observed
    # from Earth
    #
    # input: pobj = (tmin, tmix, p)
    # times at which to evaluate pobj
    # xv -- the exact positions at times times. xv.shape = (nobj, ntimes, 7)
    #
    # returns: a ndarray of maximum astrometric errors, in units of mas
    #

    # difference in true vs. approx position
    xv2 = eval_cheby(pobj, times)
    ds = np.linalg.norm(xv[:, :, 0:3] - xv2[:, :, 0:3], axis=2)
    ds = np.max(ds, axis=1)
    print(ds.shape)
    from simulation_constants import AU_M
    print(ds[:50] * AU_M)

    # position of the geocenter
    from observatory import Observatories, mjd_tai_to_et
    obs_file_path = os.path.join(basepath, OBSERVATORY_CODES)
    observer = Observatories(obs_file_path).from_obscode("500")
    oxyz = observer.barycentric(times)
    print("d_Earth ===========")
    print(oxyz.shape)
    print(oxyz[:, 0])
    print(oxyz[:, 1])
    print(oxyz[:, -1])

    # minimum topocentric distance in the interpolated range, for each object
    d = np.linalg.norm(xv[:, :, 0:3] - oxyz, axis=2)
    d = np.min(d, axis=1)
    print(d.shape)
    print(d[:50])
        
    # reduce by an Earth radius (looking for "worst case" scenarios)
    from simulation_constants import RADIUS_EARTH_KM, AU_KM
    earthR_AU = RADIUS_EARTH_KM / AU_KM
    d -= earthR_AU
    d[d < 0] = 0.  # very close approaches and impactors
    print(d[:50])

    # maximum angular error (in mas)
    alpha = np.rad2deg(ds / d) * 3600. * 1000.
    print(alpha[:50])


    return alpha

def init_integrator():
    #
    # Initializes the SPICE subsystem, and sets up the Ephem object for
    # ASSIST integrations.  The Ephem object, once set, is essentially
    # read-only and stateless (TODO: check that it really is); we therefore
    # create it only once, for all REBOUND Simulation objects to use later.
    #
    global ephem, gm_sun, gm_total

    # Furnish the spice kernels for ASSIST
    build_meta_kernel_file()
    kfn = os.path.join(basepath, META_KERNEL)
    spice.furnsh(kfn)

    # Initialize the simulation (ASSIST object) and mass constants
    ephem, gm_sun, gm_total = build_assist(basepath)

def pandas_to_ndarray(df):
    #
    # Convert a pandas.DataFrame to numpy structured array
    # with strings (or Object) columns converted to numpy fixed-width
    # strings (dtype=S).
    #
    # Returns: structured array
    #

    # inspired by https://stackoverflow.com/a/52749127
    names = df.columns
    arrays = [ df[col].values for col in names ]

    formats = [ array.dtype if array.dtype != 'O' else f'{array.astype(str).dtype}' for array in arrays ] 

    rec_array = np.rec.fromarrays( arrays, dtype={'names': names, 'formats': formats} )
    return rec_array

def approximate_ephemerides(orbits, t0, t1, order, dt = 1.):
    #
    # Input:
    #   orbits -- structured ndarray following .des naming conventions
    #   t0, t1 -- approximation range
    #   cheby_order -- the degree of chebyshev polynomials to fit
    #   dt -- sample the orbit every <dt> days
    #
    # Output:
    #   [t0, t1] -- tuple of the validity of the interpolation
    #   p -- ndarray with cheby poly fit, with p.shape = (nobj, 3, order+1)
    #   desigs -- ndarray of designations
    #   accuracy -- ndarray[f4] of approximation accuracy for Earth-based observer, in milliarcseconds
    #   xv -- ndarray of state vectors near the mid-point of the approx. interval, xv.shape = (nobj, 7)
    #         where each row is (x, y, z, vz, vy, vz, epochTDB_MJD)
    #
    # Assumes ephem, gm_sun and gm_total have been globally initialized
    #

    xv0, desig = oe_to_xv(orbits, ephem, gm_sun, gm_total)

    # compute the interpolation and test points.
    # test points will be computed at 1/2 of the dt interval
    # ensure that the beginning and the end of the interval
    # are included.
    ntimes = 2 * int(np.round((t1 - t0) / dt)) + 1
    print(f"{ntimes=}")
    times = np.linspace(t0, t1, ntimes)

    # get the samples
    xv = propagate_states(xv0, times, ephem)
    print(xv.shape)

    # fit chebyshev polynomials to every other sampling points (the
    # in-between points will be used to validate the fit), and only to the
    # spatial part of the state vector (the 0:3 part of the array)
    _, _, p = pobj = fit_cheby(times[::2], xv[:, ::2, 0:3], order, t0, t1)

    # estimate accuracy (over all points). This returns the accuracy
    # in milliarcseconds (mas).
    alpha = estimate_accuracy(pobj, times, xv)

    # extract state vectors near the middle of the interpolated interval
    xv_mid = xv[:, ntimes // 2, :]

    mask = alpha > 10
    with pd.option_context('display.float_format', '{:.2f}'.format):
#        print(pd.DataFrame(dict(desig=desig_in[mask], d=d[mask],alpha=alpha[mask], ds=ds[mask]*AU_KM)))
        print(pd.DataFrame(dict(desig=desig[mask], alpha=alpha[mask])))

    return (
        (t0, t1),
        p,
        desig,
        alpha,
        xv_mid
    )

def getobs(comps, t, observer, idx=None, maxerr_mas=10, apply_ltcorr=True, lttol_msec=10, return_ephem=False, return_lt=False):
    #
    # Compute the positions of the objects given by a ndarray
    # idx, whose return error when viewed from the Earth will be
    # not more than max_error_mas.
    #
    # t is MJD in ET, either scalar or an array (one time per object)
    #

    (t0, t1), p, desig, alpha, xv = comps
    nobj = len(desig)

    from simulation_constants import SPEED_OF_LIGHT
    assert not return_lt or apply_ltcorr, "get_obs: to return lt, apply_ltcorr must be set to True"

    # select only the requested objects
    if idx is not None:
        p, desig, alpha, xv = p[idx], desig[idx], alpha[idx], xv[idx]

    # observer barycentric position
    oxyz = np.atleast_2d(observer.barycentric(t))

    # check if there are any objects that we'll have to
    # propagate from state vectors
    svmask, = (alpha > maxerr_mas).nonzero()
    if svmask.size > 0:
        mjd_ref = ephem.jd_ref - 2400000.5
        sims = [ (i, create_sim(ephem, xv[i])) for i in svmask ]
    else:
        sims = []

    # iterate until light-time difference is <lttol_msec msec
    xyz = np.zeros((3, nobj))
    lttol = lttol_msec / 1000. / 3600. / 24.
    lt = np.zeros(nobj, dtype='f8')
    t  = t + lt # used to ensure t is a ndarray
    d  = 0.
    for _ in range(5):
        # check that the light-time correction didn't kick us out of the ephemerides validity range
        assert np.all((t0 <= t) & (t <= t1)), f"Time outside of interpolation range ({t0=}, {t1=}). {lt.max()=}, {np.max(d)=}."

        # position at t_emit. xyz.shape = (nobj, 3)
        print(p.shape)
        print(t.shape)
        xyz = eval_cheby((t0, t1, p), t, per_object_times=True)
        print(xyz.shape)
        print(xyz[:, 0:5])
        print(len(sims), desig[svmask])

        # fix up objects which have state vectors, if any
        for i, (sim, ex) in sims:
            ex.integrate_or_interpolate(t[i] - mjd_ref)
            xyz[i] = sim.particles[0].xyz
        print(xyz.shape)
        print(oxyz.shape)

        # compute topocentric position
        xyz -= oxyz
        if not apply_ltcorr:
            # exit early if we don't need light correction
            break
        print(xyz[:10])

        # distance at t_emit
        d = np.linalg.norm(xyz, axis=1)
        print(d[:10])

        # light flight time
        ltprev = lt
        lt = d / SPEED_OF_LIGHT

        print(lt[:10])
        print((lt - ltprev)[:10])

        if np.all(lt - ltprev < lttol):
            break
    else:
        assert False, "Runaway iteration in get_obs... something went wrong"

    ret = (desig, xyz)
    if return_ephem: ret += (cart_to_sph(xyz),)
    if return_lt:    ret += (lt,)
    return ret


if __name__ == "__main__":
    init_integrator()

    # convert orbital elements to barycentric state vectors
    import pandas as pd
#    orbits = pd.read_csv("mpcorb.csv").to_records(index=False)
    orbits = pandas_to_ndarray(pd.read_csv("mpcorb.csv"))
    print(orbits.dtype)
    print(orbits[0])

    t0 = 60401 # April 1st, 2024
    t1 = t0 + 10
    comps = approximate_ephemerides(orbits, t0, t1, order=7)

    # Observer
    from observatory import Observatories, mjd_tai_to_et
    obs_file_path = os.path.join(basepath, OBSERVATORY_CODES)
    observer = Observatories(obs_file_path).from_obscode("X05")

    desig, xyz, (ra, dec) = getobs(comps, t0+1.1, observer, return_ephem=True)
    exit()

    xv_in, desig_in = oe_to_xv(orbits, ephem, gm_sun, gm_total)


    # integrate to a grid of times
    t0 = 60401 # April 1st, 2024
    t1 = t0 + 30
    ntimes = 61
    times = np.linspace(t0, t1, ntimes)

    # now propagate
    xv = propagate_states(xv_in, times, ephem)
    with np.printoptions(precision=7, suppress=True, linewidth=500):
        print("xv =======")
        print(xv.shape)
        for xvobj in xv[:2]:
            print(xvobj[0])
            print(xvobj[1])
            print(xvobj[-1])
            print()


    with np.printoptions(precision=7, suppress=True, linewidth=500):
        cheby_order = 5
        (tmin, tmax, p) = pobj = fit_cheby(times[::2], xv[:, ::2], cheby_order)

#        alpha_mas = estimate_accuracy(pobj, times[1::2], xv[:, 1::2])
        alpha_mas = estimate_accuracy(pobj, times, xv)

    exit()

    with np.printoptions(precision=7, suppress=True, linewidth=500):
        # nobj, times, 7 --> times, nobj, 7 (as this is what chebfit expects)
        nobj = xv.shape[0]
        foo = np.transpose(xv, (1, 0, 2))
        print(foo.shape)
        print(foo[:, 0, 0])
        print(foo[:, 0, 1])
        print(foo[:, 0, 2])

        cheby_order = 8
        f = np.linspace(0, 1, ntimes)
        p = np.polynomial.chebyshev.chebfit(f, foo.reshape(ntimes, -1), cheby_order).reshape(cheby_order+1, nobj, -1)
        # transpose to (nobj, 7, cheby+1) shape, to keep all the data belonging to the same object
        # close by in memory (and on disk). This will be our canonical storage format.
        p = np.transpose(p, (1, 2, 0))
        print("p =======")
        print(p.shape)
        print(p[0])

        # evaluate chebys, print state vectors for a few times for the 0th object
        xv2 = np.polynomial.chebyshev.chebval(f, p.transpose(2, 0, 1)).transpose(0, 2, 1)
        print("xv2 =======")
        print(xv2.shape) # (nobj, ntimes, 7)
        print(xv2[0,  0])
        print(xv2[0,  1])
        print(xv2[0, -1])

        # compute the differences between computed and decompressed
        foo = np.linalg.norm(xv[:, :, 0:3] - xv2[:, :, 0:3], axis=2)
        foo = np.max(foo, axis=1)
        print(foo.shape)
        from simulation_constants import AU_M
        print(foo[:50] * AU_M)

#        print(xv2.shape) # (nobj, 7, ntimes)
#        print(xv2[0, :, 0])
#        print(xv2[0, :, 1])
#        print(xv2[0, :, -1])

    # now compress to chebys
#    p = compress(xv)
