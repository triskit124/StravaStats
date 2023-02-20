import math
import numpy as np

def llh_to_ecef(lat: float, lon: float, h: float):
    """
    Converts lat, lon, height to Earth-centered, Earth-fixed coordinates
    lat : latitude [deg]
    lon : longitude [deg]
    h : height [m]
    """

    a = 6378137.0; # WGS84 Semimajor axis [m]
    b = 6356752.314245; # WGS84 Semiminor axis [m]

    lat = np.radians(lat)
    lon = np.radians(lon)

    Nphi = (a**2)/math.sqrt(a**2 * math.cos(lat)**2 + b**2 * math.sin(lat)**2)
    x = (Nphi + h) * math.cos(lat) * math.cos(lon)
    y = (Nphi + h) * math.cos(lat) * math.sin(lon)
    z = ((b**2 / a**2) * Nphi + h) * math.sin(lat)

    return x, y, z


def ecef_to_enu(x: float, y: float, z: float, lat: float, lon: float, height: float):
    """
    Converts Earth-centered, Earth-fixed coordinates (ECEF) of a point of interest (x, y, z) into 
    a local East North Up (ENU) frame located at (lat, long, height). 

    x : ECEF coordinate of point of interest [m]
    y : ECEF coordinate of point of interest [m]
    z : ECEF coordinate of point of interest [m]
    lat: latitude of origin [deg]
    lon: longitude of origin [deg]
    """

    # compute ECEF coordinates of local ENU frame location
    x0, y0, z0 = llh_to_ecef(lat, lon, height)

    lat = np.radians(lat)
    lon = np.radians(lon)

    # take coordinate w.r.t frame origin
    x -= x0
    y -= y0
    z -= z0

    east = x * (-math.sin(lon)) + y * (math.cos(lon))
    north = x * (-math.sin(lat) * math.cos(lon)) + y * (-math.sin(lat) * math.sin(lon)) + z * (math.cos(lat))
    up = x * (math.cos(lat) * math.cos(lon)) + y * (math.cos(lat) * math.sin(lon)) + z * (math.sin(lat))

    return east, north, up