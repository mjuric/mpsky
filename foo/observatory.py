import json
import os
import numpy as np
import spiceypy as spice

from simulation_constants import RADIUS_EARTH_KM, AU_KM
from simulation_geometry import ecliptic_to_equatorial
from orbit_conversion_utilities import universal_cartesian

def mjd_tai_to_et(mjd_tai):
    """
    Converts a MJD value in TAI to ET
    """
    return mjd_tai + 32.184 / 86400.

class Observatories:
    _obsdata = None # internal data with observatories

    def _convert_to_geocentric(self, obs_location):
        # convert to geocentric vector (in km), returning
        # None-tuples for observatories w/o constants (typically
        # spacecraft/roving observers).

        if (
            obs_location.get("Longitude", False)
            and obs_location.get("cos", False)
            and obs_location.get("sin", False)
        ):
            longitude = obs_location["Longitude"] * np.pi / 180.0
            x = obs_location["cos"] * np.cos(longitude)
            y = obs_location["cos"] * np.sin(longitude)
            z = obs_location["sin"]
            return (x, y, z)
        else:
            return (None, None, None)

    def __init__(self, obs_file_path):
        # load 
        with open(obs_file_path) as fp:
            obs = json.load(fp)

        self._obsdata = { obs_name: self._convert_to_geocentric(obs_loc) for obs_name, obs_loc in obs.items() }

    def from_obscode(self, obscode):
        return Observer(self._obsdata[obscode])

class Observer:
    # Get the MPC's unit vector from the geocenter to the observatory
    _obsvec = None

    # NOTE: Do not construct directly -- use Observatories.from_obscode()
    def __init__(self, obsvec):
        self.obsvec = obsvec

    def _barycentric_aux(self, mjd_et):
        assert mjd_et < 2400000.5, "Time must be in MJD (TDB)"
        mjd2000 = spice.j2000() - 2400000.5
        et_s = (mjd_et - mjd2000) * 24 * 60 * 60

        # Get the barycentric position of Earth
        pos, _ = spice.spkpos("EARTH", et_s, "J2000", "NONE", "SSB")

        # Get the matrix that rotates from the Earth's equatorial body fixed frame to the J2000 equatorial frame.
        m = spice.pxform("ITRF93", "J2000", et_s)

        # Carry out the rotation and scale
        mVec = np.dot(m, self.obsvec) * RADIUS_EARTH_KM

#        print(f"{(et_s, pos/AU_KM, mVec/AU_KM)=}")
#        exit()
        return (pos + mVec) / AU_KM

    def barycentric(self, mjd_et):
        """
        Computes the barycentric position of the observer

        Parameters
        ----------
            mjd_et : float or array
                MJD (in ET)
        Returns
        -------
            : array (3, ) or (3, len(et))
                Barycentric position of the observatory (x,y,z), in AU
        """

        if np.isscalar(mjd_et):
            out = self._barycentric_aux(mjd_et)
        else:
            out = np.empty((len(mjd_et), 3))
            for k, mjd_ett in enumerate(mjd_et):
                out[k, :] = self._barycentric_aux(mjd_ett)
        return out

if __name__ == "__main__":
    basepath = "/Users/mjuric/Library/Caches/sorcha"
    META_KERNEL = "meta_kernel.txt"

    kfn = os.path.join(basepath, META_KERNEL)
    spice.furnsh(kfn)

    obs_file_path = os.path.join(basepath, OBSERVATORY_CODES)
    obs = Observatories(obs_file_path).from_obscode("X05")

    import time
    t0 = time.perf_counter()
    nsamples = 100_000

    # On an M1 AppleSilicon this takes ~0.01 msec
    for _ in range(nsamples):
        et = mjd_tai_to_et(60232) + 2400000.5
        pos = obs.barycentric(et)
    dt = time.perf_counter() - t0

    print(pos)
    print(f"exec_time: {dt/nsamples*1000:.4} msec")
