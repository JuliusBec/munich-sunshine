import duckdb
import httpx
import threading
from datetime import datetime, timezone

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from shapely import wkb

from shadows import compute_shadows, sun_position, sunshine_remaining, get_daylight_times


app = FastAPI(title="Munich Sunshine")

# ---------------------------------------------------------------------------
# Thread-local DB connections
# DuckDB connections are not thread-safe. FastAPI runs sync endpoints in a
# thread pool, so we give each thread its own read-only connection.
# ---------------------------------------------------------------------------

_local = threading.local()

def _db() -> duckdb.DuckDBPyConnection:
    if not hasattr(_local, "con"):
        _local.con = duckdb.connect("munich.duckdb", read_only=True)
        _local.con.execute("LOAD spatial;")
    return _local.con


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_dt(dt_str: str | None) -> datetime:
    """Parse ISO-8601 string or fall back to current UTC time."""
    if dt_str is None:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {dt_str!r}")


def _query_buildings_near(db, lat: float, lon: float, radius_m: float = 400) -> list[dict]:
    """Return buildings within radius_m metres of (lat, lon)."""
    # At 48°N: 1° lat ≈ 111 km, 1° lon ≈ 74 km — use lat degrees as conservative radius
    radius_deg = radius_m / 111_000
    rows = db.execute("""
        SELECT ST_AsWKB(geom), name, resolved_height
        FROM buildings
        WHERE ST_DWithin(geom, ST_Point(?, ?), ?)
    """, [lon, lat, radius_deg]).fetchall()
    return [
        {"geom": wkb.loads(bytes(row[0])), "name": row[1], "resolved_height": row[2]}
        for row in rows
    ]


def _query_buildings(db, west: float, south: float, east: float, north: float) -> list[dict]:
    """Return buildings whose geometry intersects the given bounding box."""
    rows = db.execute("""
        SELECT
            ST_AsWKB(geom),
            name,
            resolved_height
        FROM buildings
        WHERE ST_Intersects(
            geom,
            ST_MakeEnvelope(?, ?, ?, ?)
        )
    """, [west, south, east, north]).fetchall()

    return [
        {"geom": wkb.loads(bytes(row[0])), "name": row[1], "resolved_height": row[2]}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/sun")
def get_sun(dt: str | None = Query(None, description="ISO-8601 UTC datetime")):
    """Current (or requested) sun position over Munich + cloud cover."""
    parsed_dt = _parse_dt(dt)
    azimuth, elevation = sun_position(parsed_dt)

    cloud_cover = None
    try:
        resp = httpx.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": 48.1374,
            "longitude": 11.5755,
            "current": "cloud_cover",
        }, timeout=5)
        cloud_cover = resp.json()["current"]["cloud_cover"]
    except Exception:
        pass  # non-fatal — frontend can display shadows without cloud info

    daylight = get_daylight_times(parsed_dt)

    return {
        "datetime": parsed_dt.isoformat(),
        "azimuth": azimuth,
        "elevation": elevation,
        "above_horizon": elevation > 0,
        "cloud_cover": cloud_cover,
        "sunrise": daylight["sunrise"],
        "sunset": daylight["sunset"],
    }


@app.get("/api/shadows")
def get_shadows(
    west:  float = Query(..., description="Bounding box west longitude"),
    south: float = Query(..., description="Bounding box south latitude"),
    east:  float = Query(..., description="Bounding box east longitude"),
    north: float = Query(..., description="Bounding box north latitude"),
    dt:    str | None = Query(None, description="ISO-8601 UTC datetime (default: now)"),
):
    """
    Return shadow polygons for all buildings in the given bbox at the given time.
    Response is a GeoJSON FeatureCollection with an extra 'sun' and 'merged_shadow' field.
    """
    # Guard against huge requests
    if (east - west) > 0.2 or (north - south) > 0.2:
        raise HTTPException(status_code=400, detail="Bounding box too large (max ~15 km)")

    parsed_dt = _parse_dt(dt)
    buildings = _query_buildings(_db(), west, south, east, north)
    result = compute_shadows(buildings, parsed_dt)
    result["building_count"] = len(buildings)
    return result


@app.get("/api/sunshine")
def get_sunshine(
    lat: float = Query(..., description="Latitude of the point to check"),
    lon: float = Query(..., description="Longitude of the point to check"),
    dt:  str | None = Query(None, description="ISO-8601 UTC datetime (default: now)"),
):
    """
    For a given point, return whether it's currently sunny and how long that will last.
    Queries nearby buildings (within 400 m) and steps forward through time to find
    the next shadow/sun transition.
    """
    parsed_dt = _parse_dt(dt)
    buildings = _query_buildings_near(_db(), lat, lon)
    result = sunshine_remaining(lat, lon, buildings, parsed_dt)
    result["lat"] = lat
    result["lon"] = lon
    result["datetime"] = parsed_dt.isoformat()
    result["nearby_buildings"] = len(buildings)
    return result


@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html") as f:
        return f.read()
