"""
Shadow polygon computation.
Ported from the single-building proof-of-concept in shadow.py to work on
a list of buildings returned from DuckDB.
"""

import numpy as np
import pvlib
import pandas as pd
from shapely.geometry import Point, Polygon, mapping
from shapely.affinity import translate
from shapely.ops import unary_union
from datetime import datetime, timedelta, timezone


def sun_position(dt: datetime) -> tuple[float, float]:
    """Return (azimuth_deg, elevation_deg) for Munich at the given UTC datetime."""
    lat, lon = 48.1374, 11.5755
    times = pd.DatetimeIndex([dt])
    solar = pvlib.solarposition.get_solarposition(times, lat, lon)
    azimuth = float(solar["azimuth"].iloc[0])
    elevation = float(solar["apparent_elevation"].iloc[0])
    return azimuth, elevation


# Approximate metric scale at Munich's latitude (48°N)
_M_PER_DEG_LAT = 111_000
_M_PER_DEG_LON = 111_000 * np.cos(np.radians(48.1374))

# Cap shadow length to avoid absurdly long shadows at sunrise/sunset.
# At 5° elevation tan(5°) ≈ 0.087, so a 10m building casts a ~115m shadow.
# At 2° it becomes ~286m. We cap at 300m worth of degrees.
_MAX_SHADOW_M = 300


def _extrude_polygon_shadow(poly: Polygon, d_lon: float, d_lat: float):
    """
    Compute the shadow of a single Polygon by extruding each outward-facing edge
    in the shadow direction. Produces an exact shadow (no gap at the base,
    no false infill of concave areas like courtyards).
    """
    pieces = [poly]
    coords = list(poly.exterior.coords)
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        # Outward normal for a CCW exterior ring: rotate edge vector 90° CW → (y2-y1, x1-x2)
        # Extrude this edge if the shadow direction has a positive component along the normal.
        if d_lon * (y2 - y1) + d_lat * (x1 - x2) > 0:
            quad = Polygon([
                (x1, y1), (x2, y2),
                (x2 + d_lon, y2 + d_lat),
                (x1 + d_lon, y1 + d_lat),
            ])
            pieces.append(quad.convex_hull)
    return unary_union(pieces)


def _shadow_polygon(footprint, height: float, azimuth: float, elevation: float):
    """
    Project a single building footprint into a shadow polygon.
    Returns a Shapely geometry or None if the sun is below the horizon.
    """
    if elevation <= 0 or height <= 0:
        return None

    shadow_length = min(height / np.tan(np.radians(elevation)), _MAX_SHADOW_M)
    d_lon = -shadow_length * np.sin(np.radians(azimuth)) / _M_PER_DEG_LON
    d_lat = -shadow_length * np.cos(np.radians(azimuth)) / _M_PER_DEG_LAT

    if footprint.geom_type == "Polygon":
        return _extrude_polygon_shadow(footprint, d_lon, d_lat)
    elif footprint.geom_type == "MultiPolygon":
        return unary_union([_extrude_polygon_shadow(p, d_lon, d_lat) for p in footprint.geoms])
    else:
        # Fallback for other geometry types (LineString, etc.)
        return footprint.union(translate(footprint, xoff=d_lon, yoff=d_lat))


def compute_shadows(buildings: list[dict], dt: datetime) -> dict:
    """
    Compute merged shadow polygons for a list of buildings at a given time.

    Each building dict must have:
        geom            – Shapely geometry (WGS-84 lon/lat)
        resolved_height – float, metres

    Returns a GeoJSON FeatureCollection with:
        - one feature per building shadow (for debugging / per-building info)
        - a 'merged' property on the collection with the unioned shadow polygon
    """
    azimuth, elevation = sun_position(dt)

    if elevation <= 0:
        return {
            "type": "FeatureCollection",
            "sun": {"azimuth": azimuth, "elevation": elevation, "above_horizon": False},
            "features": [],
        }

    features = []
    shadow_geoms = []

    for b in buildings:
        footprint = b["geom"]
        height = b["resolved_height"] or 0
        shadow = _shadow_polygon(footprint, height, azimuth, elevation)
        if shadow is None or shadow.is_empty:
            continue
        shadow_geoms.append(shadow)
        features.append({
            "type": "Feature",
            "geometry": mapping(shadow),
            "properties": {
                "name": b.get("name"),
                "height": height,
            },
        })

    merged = unary_union(shadow_geoms) if shadow_geoms else None

    return {
        "type": "FeatureCollection",
        "sun": {"azimuth": azimuth, "elevation": elevation, "above_horizon": True},
        "merged_shadow": mapping(merged) if merged else None,
        "features": features,
    }


def project_to_building_exterior(lat: float, lon: float, buildings: list[dict], offset_m: float = 3.0):
    """
    If (lat, lon) falls inside a building footprint, return a point offset_m metres
    outside the nearest edge (outward from the building centroid).
    Returns (lat, lon) unchanged if the point isn't inside any building.
    """
    pt = Point(lon, lat)
    for b in buildings:
        if b["geom"].contains(pt):
            exterior = b["geom"].exterior if b["geom"].geom_type == "Polygon" else b["geom"].geoms[0].exterior
            # nearest point on the boundary
            nearest = exterior.interpolate(exterior.project(pt))
            centroid = b["geom"].centroid
            # vector from centroid to boundary point, in metres
            dx_m = (nearest.x - centroid.x) * _M_PER_DEG_LON
            dy_m = (nearest.y - centroid.y) * _M_PER_DEG_LAT
            dist_m = (dx_m ** 2 + dy_m ** 2) ** 0.5
            if dist_m == 0:
                break
            new_lat = nearest.y + (dy_m / dist_m) * offset_m / _M_PER_DEG_LAT
            new_lon = nearest.x + (dx_m / dist_m) * offset_m / _M_PER_DEG_LON
            return new_lat, new_lon
    return lat, lon


def _is_in_shadow(lat: float, lon: float, buildings: list[dict], azimuth: float, elevation: float) -> bool:
    """Return True if the point (lat, lon) falls inside any building's shadow."""
    pt = Point(lon, lat)
    for b in buildings:
        shadow = _shadow_polygon(b["geom"], b["resolved_height"] or 0, azimuth, elevation)
        if shadow and pt.within(shadow):
            return True
    return False


def get_daylight_times(dt: datetime) -> dict:
    """Return sunrise and sunset as UTC ISO strings for the day containing dt."""
    lat, lon = 48.1374, 11.5755
    date_str = dt.strftime('%Y-%m-%d')
    times = pd.date_range(start=date_str, periods=288, freq='5min', tz='UTC')
    solar = pvlib.solarposition.get_solarposition(times, lat, lon)
    elev = solar['apparent_elevation']

    sunrise = sunset = None
    for i in range(1, len(elev)):
        if elev.iloc[i - 1] <= 0 and elev.iloc[i] > 0 and sunrise is None:
            sunrise = times[i].isoformat()
        elif elev.iloc[i - 1] > 0 and elev.iloc[i] <= 0 and sunset is None:
            sunset = times[i - 1].isoformat()
    return {"sunrise": sunrise, "sunset": sunset}


def sunshine_remaining(lat: float, lon: float, buildings: list[dict], dt: datetime) -> dict:
    """
    From dt, step forward in 5-minute increments to find the next sunshine
    state change at (lat, lon).

    Returns:
        currently_sunny  – bool
        minutes          – minutes until state change (or until sunset if sunny,
                           None if in shadow until sunset)
        changes_at       – ISO-8601 string of the transition moment, or None
        reason           – 'shadow' | 'sunset' | 'sun' | 'shadow_until_sunset' | 'sunny_all_day'
    """
    STEP = 5  # minutes per iteration

    az, el = sun_position(dt)
    if el <= 0:
        return {"currently_sunny": False, "minutes": None, "changes_at": None, "reason": "night"}

    currently_sunny = not _is_in_shadow(lat, lon, buildings, az, el)

    for i in range(1, 200):  # max ~16 hours forward
        t = dt + timedelta(minutes=STEP * i)
        az, el = sun_position(t)

        if el <= 0:
            # Reached sunset
            if currently_sunny:
                return {"currently_sunny": True,  "minutes": STEP * i, "changes_at": t.isoformat(), "reason": "sunset"}
            else:
                return {"currently_sunny": False, "minutes": None,      "changes_at": None,           "reason": "shadow_until_sunset"}

        sunny_now = not _is_in_shadow(lat, lon, buildings, az, el)
        if sunny_now != currently_sunny:
            return {
                "currently_sunny": currently_sunny,
                "minutes": STEP * i,
                "changes_at": t.isoformat(),
                "reason": "shadow" if currently_sunny else "sun",
            }

    return {"currently_sunny": currently_sunny, "minutes": None, "changes_at": None, "reason": "sunny_all_day"}
