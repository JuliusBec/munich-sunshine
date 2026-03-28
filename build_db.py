import json
import tempfile
import os
import duckdb
import osmium

# ── Munich area bounding box (used to filter parts from Oberbayern PBF) ────
MUNICH_BBOX = (11.3, 47.9, 11.9, 48.4)  # west, south, east, north

# Munich city center (Marienplatz) — used for inner/outer district split
CENTER_LON = 11.5755
CENTER_LAT = 48.1374
INNER_RADIUS_DEG = 3000 / 111_000  # 3 km in degrees (lat)

# Per-type height defaults (meters) for buildings with no height or levels tag.
# Inner district default: 4 levels × 3m = 12m  (dense Altbau / Gründerzeit)
# Outer district default: 2 levels × 3m =  6m  (houses, suburbs)
INNER_DEFAULT_M = 12.0
OUTER_DEFAULT_M = 6.0

# Explicit per-type overrides that apply everywhere regardless of zone.
# These building types are structurally distinct enough to warrant their own default.
TYPE_DEFAULTS = {
    "garage":           3.0,
    "garages":          3.0,
    "shed":             3.0,
    "barn":             4.0,
    "farm_auxiliary":   4.0,
    "roof":             3.0,
    "allotment_house":  3.0,
    "kiosk":            3.0,
    "stadium":         15.0,
    "grandstand":      12.0,
    "bleachers":       10.0,
}

# Name-based overrides for landmarks where OSM height/levels data is inaccurate.
# Only needed for buildings that lack detailed building:part mapping in OSM.
# Priority: name override > OSM explicit height > levels > type default > zone default.
NAME_OVERRIDES = {
    "Neues Rathaus":                                    85.0,  # Rathausturm 85m
    "Frauenkirche":                                     99.0,  # twin towers ~99m
    "Allianz Arena":                                    50.0,  # outer facade ~48m
    "Städtisches Stadion an der Grünwalder Straße":     14.0,  # stands ~12–15m
}


# ── Step 1: Extract building:part ways from Oberbayern PBF ─────────────────

class _PartExtractor(osmium.SimpleHandler):
    def __init__(self, bbox):
        super().__init__()
        self.bbox = bbox  # west, south, east, north
        self.features = []

    def way(self, w):
        if "building:part" not in w.tags:
            return
        try:
            coords = [(n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            return
        if len(coords) < 4:
            return
        # Ensure ring is closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        west, south, east, north = self.bbox
        avg_lon = sum(c[0] for c in coords) / len(coords)
        avg_lat = sum(c[1] for c in coords) / len(coords)
        if not (west <= avg_lon <= east and south <= avg_lat <= north):
            return

        self.features.append({
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": dict(w.tags),
        })


def extract_building_parts(pbf_path, bbox):
    print(f"Extracting building:part ways from {pbf_path}…")
    extractor = _PartExtractor(bbox)
    extractor.apply_file(pbf_path, locations=True)
    print(f"  Found {len(extractor.features):,} building:part ways in Munich area")
    return extractor.features


# ── Height resolution SQL (shared between outlines and parts) ───────────────

def _height_sql(table):
    name_cases = "\n".join(
        f"        WHEN name = '{n}' THEN {h}" for n, h in NAME_OVERRIDES.items()
    )
    type_cases = "\n".join(
        f"        WHEN building_type = '{t}' THEN {h}" for t, h in TYPE_DEFAULTS.items()
    )
    type_list = ", ".join(f"'{t}'" for t in TYPE_DEFAULTS)
    return f"""
    ALTER TABLE {table} ADD COLUMN resolved_height DOUBLE;
    UPDATE {table} SET resolved_height = CASE
{name_cases}
        WHEN height IS NOT NULL
            THEN height
        WHEN levels IS NOT NULL AND levels > 0
            THEN levels * 3.0
        WHEN building_type IN ({type_list})
            THEN CASE
{type_cases}
                 ELSE {OUTER_DEFAULT_M}
                 END
        WHEN ST_Within(geom, ST_Buffer(ST_Point({CENTER_LON}, {CENTER_LAT}), {INNER_RADIUS_DEG}))
            THEN {INNER_DEFAULT_M}
        ELSE
            {OUTER_DEFAULT_M}
    END;
"""


# ── Main build ──────────────────────────────────────────────────────────────

con = duckdb.connect("munich.duckdb")
con.execute("INSTALL spatial; LOAD spatial;")

# 1. Load building outlines
print("Loading building outlines from GeoJSONL…")
con.execute("""
    CREATE OR REPLACE TABLE outlines AS
    SELECT
        ST_GeomFromGeoJSON(geometry::TEXT)               AS geom,
        properties->>'name'                              AS name,
        properties->>'building'                          AS building_type,
        TRY_CAST(properties->>'height'         AS DOUBLE) AS height,
        TRY_CAST(properties->>'building:levels' AS INT)   AS levels
    FROM read_json(
        'munich_buildings_plain.geojsonl',
        format='newline_delimited',
        columns={geometry: 'JSON', properties: 'JSON'}
    )
""")
con.execute(_height_sql("outlines"))
print(f"  {con.execute('SELECT COUNT(*) FROM outlines').fetchone()[0]:,} outlines loaded")

# 2. Extract and load building:parts
parts = extract_building_parts("oberbayern.osm.pbf", MUNICH_BBOX)

with tempfile.NamedTemporaryFile(mode="w", suffix=".geojsonl", delete=False) as f:
    parts_path = f.name
    for feat in parts:
        f.write(json.dumps(feat) + "\n")

con.execute(f"""
    CREATE OR REPLACE TABLE parts AS
    SELECT
        ST_GeomFromGeoJSON(geometry::TEXT)                    AS geom,
        properties->>'name'                                   AS name,
        COALESCE(properties->>'building:part',
                 properties->>'building')                     AS building_type,
        TRY_CAST(properties->>'height'         AS DOUBLE)    AS height,
        TRY_CAST(properties->>'building:levels' AS INT)       AS levels
    FROM read_json(
        '{parts_path}',
        format='newline_delimited',
        columns={{geometry: 'JSON', properties: 'JSON'}}
    )
""")
con.execute(_height_sql("parts"))
os.unlink(parts_path)
print(f"  {con.execute('SELECT COUNT(*) FROM parts').fetchone()[0]:,} parts loaded")

# 3. Identify outlines substantially covered by parts (> 30% area overlap).
#    These will be replaced by their parts so we don't double-count.
print("Identifying outlines replaced by parts…")
con.execute("""
    CREATE OR REPLACE TABLE outlines_with_parts AS
    SELECT DISTINCT o.rowid
    FROM outlines o
    JOIN parts p ON ST_Intersects(o.geom, p.geom)
    WHERE ST_IsValid(o.geom) AND ST_IsValid(p.geom)
      AND o.height IS NULL   -- keep outlines that already have accurate explicit height tags
      AND p.resolved_height > 5  -- ignore roof/shed parts that don't represent a main body
      AND ST_Area(ST_Intersection(o.geom, p.geom)) / NULLIF(ST_Area(o.geom), 0) > 0.3
""")
replaced = con.execute("SELECT COUNT(*) FROM outlines_with_parts").fetchone()[0]
print(f"  {replaced:,} outlines replaced by parts")

# 4. Build final buildings table: parts + outlines not covered by parts
print("Building final buildings table…")
con.execute("""
    CREATE OR REPLACE TABLE buildings AS
    SELECT geom, name, building_type, height, levels, resolved_height FROM parts
    UNION ALL
    SELECT geom, name, building_type, height, levels, resolved_height FROM outlines
    WHERE rowid NOT IN (SELECT rowid FROM outlines_with_parts)
""")
con.execute("DROP TABLE outlines; DROP TABLE parts; DROP TABLE outlines_with_parts;")
total_buildings = con.execute("SELECT COUNT(*) FROM buildings").fetchone()[0]
print(f"  {total_buildings:,} buildings in final table")

# ── Summary ─────────────────────────────────────────────────────────────────

print("\nHeight resolution summary:")
rows = con.execute("""
    SELECT
        CASE
            WHEN name IN (""" + ", ".join(f"'{n}'" for n in NAME_OVERRIDES) + """)
                                                           THEN '0. name override'
            WHEN height IS NOT NULL                        THEN '1. explicit height tag'
            WHEN levels IS NOT NULL AND levels > 0         THEN '2. levels × 3m'
            WHEN building_type IN ('garage','garages','shed','barn','farm_auxiliary',
                                   'roof','allotment_house','kiosk','stadium',
                                   'grandstand','bleachers')
                                                           THEN '3. type override'
            ELSE                                               '4. zone default'
        END AS source,
        COUNT(*) AS n
    FROM buildings
    GROUP BY 1 ORDER BY 1
""").fetchall()
total = sum(r[1] for r in rows)
for source, n in rows:
    print(f"  {source}: {n:,} ({n/total*100:.1f}%)")

print("\nSpot checks:")
for name in ['Neues Rathaus', 'Frauenkirche', 'BMW-Vierzylinder', 'Olympiaturm',
             'Allianz Arena', 'Städtisches Stadion an der Grünwalder Straße']:
    rows = con.execute("""
        SELECT name, building_type, height, levels, resolved_height
        FROM buildings WHERE name = ?
        ORDER BY resolved_height DESC LIMIT 1
    """, [name]).fetchall()
    if rows:
        r = rows[0]
        print(f"  {r[0]}: type={r[1]}, height={r[2]}, levels={r[3]} → resolved={r[4]}m")
    else:
        print(f"  {name}: NOT FOUND")

con.close()
print("\nSaved to munich.duckdb")
