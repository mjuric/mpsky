@numba.njit
def oe_to_state(orbits, epochJD_TDB, ephem, sun_dict, gm_sun, gm_total):
    # Convert orbital elements to state vector

    # initialize output arrays
    sv = np.zeros((6, len(orbits))
    x, y, z, vx, vy, vz = sv

    for i, row in enumerate(orbits):
        orbit_format = row["FORMAT"]

        if orbit_format in ["CART", "BCART"]:
            ecx, ecy, ecz = row["x"], row["y"], row["z"]
            dx, dy, dz = row["xdot"], row["ydot"], row["zdot"]
        elif orbit_format == "COM":
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

    if epochJD_TDB not in sun_dict:
        sun_dict[epochJD_TDB] = ephem.get_particle("Sun", epochJD_TDB - ephem.jd_ref)

    sun = sun_dict[epochJD_TDB]

    equatorial_coords = np.array(ecliptic_to_equatorial([ecx, ecy, ecz]))
    equatorial_velocities = np.array(ecliptic_to_equatorial([dx, dy, dz]))

    if orbit_format in ["KEP", "COM", "CART"]:
        equatorial_coords += np.array((sun.x, sun.y, sun.z))
        equatorial_velocities += np.array((sun.vx, sun.vy, sun.vz))

    return tuple(np.concatenate([equatorial_coords, equatorial_velocities]))

