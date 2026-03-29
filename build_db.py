import json
import tempfile
import os
import shutil
import duckdb
import osmium
from shapely.geometry import LineString, mapping
from shapely.ops import linemerge, polygonize, unary_union

# Munich area bounding box (used to filter from Oberbayern PBF)
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

# Name-based height overrides for landmarks where OSM height/levels data is
# absent or inaccurate. Applied to both outlines and relations.
NAME_OVERRIDES = {
    "Neues Rathaus":    85.0,  # Rathausturm 85m
    "Frauenkirche":     99.0,  # twin towers ~99m
}


# ── Pass 1: scan relations to find building relations and their member way IDs ─

class _RelationScanner(osmium.SimpleHandler):
    """Collect metadata and member way IDs for all building=* relations."""
    def __init__(self):
        super().__init__()
        self.relations   = {}         # rel_id → {tags, outers, inners}
        self.member_ways = set()      # all way IDs referenced by those relations

    def relation(self, r):
        if "building" not in r.tags:
            return
        outers, inners = [], []
        for m in r.members:
            if m.type != "w":
                continue
            if m.role == "outer":
                outers.append(m.ref)
            elif m.role == "inner":
                inners.append(m.ref)
        if not outers:
            return
        self.relations[r.id] = {
            "tags":   dict(r.tags),
            "outers": outers,
            "inners": inners,
        }
        self.member_ways.update(outers)
        self.member_ways.update(inners)


# ── Pass 2: extract building:part ways AND member way geometries in one pass ──

class _WayExtractor(osmium.SimpleHandler):
    """
    Single location-aware pass over the PBF that does two things:
      1. Collects building:part way features (for shadow detail on landmarks).
      2. Collects raw coordinates for relation member ways (for ring assembly).
    """
    def __init__(self, bbox, relation_way_ids):
        super().__init__()
        self.bbox             = bbox
        self.relation_way_ids = relation_way_ids
        self.part_features    = []          # GeoJSON-like dicts
        self.way_coords       = {}          # way_id → [(lon, lat), …]

    def way(self, w):
        try:
            coords = [(n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            return

        # Always store if it is a relation member (needed for ring stitching)
        if w.id in self.relation_way_ids and len(coords) >= 2:
            self.way_coords[w.id] = coords

        # Collect building:part ways inside the Munich bbox
        if "building:part" in w.tags and len(coords) >= 4:
            if coords[0] != coords[-1]:
                coords = coords + [coords[0]]
            west, south, east, north = self.bbox
            avg_lon = sum(c[0] for c in coords) / len(coords)
            avg_lat = sum(c[1] for c in coords) / len(coords)
            if west <= avg_lon <= east and south <= avg_lat <= north:
                self.part_features.append({
                    "geometry":   {"type": "Polygon", "coordinates": [coords]},
                    "properties": dict(w.tags),
                })


# ── Geometry assembly for relations ──────────────────────────────────────────

def _assemble_relation_geom(outer_ids, inner_ids, way_coords):
    """
    Stitch outer and inner way IDs into a Shapely polygon (with holes).
    Multiple ways forming one ring are merged via linemerge + polygonize.
    Returns a Shapely geometry or None.
    """
    def ids_to_polys(ids):
        lines = [
            LineString(way_coords[wid])
            for wid in ids
            if wid in way_coords and len(way_coords[wid]) >= 2
        ]
        if not lines:
            return []
        return list(polygonize(linemerge(lines)))

    outer_polys = ids_to_polys(outer_ids)
    if not outer_polys:
        return None

    inner_polys = ids_to_polys(inner_ids)

    parts = []
    for outer in outer_polys:
        holes = [inn for inn in inner_polys if outer.contains(inn.centroid)]
        parts.append(outer.difference(unary_union(holes)) if holes else outer)

    return unary_union(parts) if len(parts) > 1 else parts[0]


# ── Top-level extraction function ────────────────────────────────────────────

def extract_from_pbf(pbf_path, bbox):
    """
    Two-pass extraction from the PBF:
      Pass 1 — scan relations (no location data needed).
      Pass 2 — extract way geometries (location-aware, single pass).

    Returns (part_features, relation_features) as lists of GeoJSON-like dicts.
    """
    west, south, east, north = bbox

    print(f"PBF pass 1: scanning building relations…")
    scanner = _RelationScanner()
    scanner.apply_file(pbf_path)
    print(f"  {len(scanner.relations):,} building relations, "
          f"{len(scanner.member_ways):,} member ways")

    print(f"PBF pass 2: extracting ways (parts + relation members)…")
    extractor = _WayExtractor(bbox, scanner.member_ways)
    extractor.apply_file(pbf_path, locations=True)
    print(f"  {len(extractor.part_features):,} building:part ways in bbox")
    print(f"  {len(extractor.way_coords):,} relation member way geometries collected")

    # Assemble relation features, filtered to bbox
    relation_features = []
    for rel in scanner.relations.values():
        # Quick bbox check on first outer way
        first = next(
            (extractor.way_coords[wid] for wid in rel["outers"]
             if wid in extractor.way_coords),
            None,
        )
        if first is None:
            continue
        avg_lon = sum(c[0] for c in first) / len(first)
        avg_lat = sum(c[1] for c in first) / len(first)
        if not (west <= avg_lon <= east and south <= avg_lat <= north):
            continue

        geom = _assemble_relation_geom(
            rel["outers"], rel["inners"], extractor.way_coords
        )
        if geom is None or geom.is_empty:
            continue

        tags = rel["tags"]
        relation_features.append({
            "geometry":   mapping(geom),
            "properties": {
                "name":             tags.get("name"),
                "building":         tags.get("building"),
                "height":           tags.get("height"),
                "building:levels":  tags.get("building:levels"),
            },
        })

    print(f"  {len(relation_features):,} building relations assembled in bbox")
    return extractor.part_features, relation_features


# ── Height resolution SQL ────────────────────────────────────────────────────

def _height_sql(table, include_name_overrides=True):
    name_cases = ""
    if include_name_overrides:
        name_cases = "\n".join(
            f"        WHEN name = '{n}' THEN {h}" for n, h in NAME_OVERRIDES.items()
        ) + "\n"
    type_cases = "\n".join(
        f"        WHEN building_type = '{t}' THEN {h}" for t, h in TYPE_DEFAULTS.items()
    )
    type_list = ", ".join(f"'{t}'" for t in TYPE_DEFAULTS)
    return f"""
    ALTER TABLE {table} ADD COLUMN resolved_height DOUBLE;
    UPDATE {table} SET resolved_height = CASE
{name_cases}        WHEN height IS NOT NULL
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


def _write_temp_geojsonl(features):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".geojsonl", delete=False)
    for feat in features:
        f.write(json.dumps(feat) + "\n")
    f.close()
    return f.name


# ── Main build ──────────────────────────────────────────────────────────────

con = duckdb.connect("munich.duckdb")
con.execute("INSTALL spatial; LOAD spatial;")

# 1. Load building outlines from GeoJSONL (way-based buildings)
print("Loading building outlines from GeoJSONL…")
con.execute("""
    CREATE OR REPLACE TABLE outlines AS
    SELECT
        ST_GeomFromGeoJSON(geometry::TEXT)                AS geom,
        properties->>'name'                               AS name,
        properties->>'building'                           AS building_type,
        TRY_CAST(properties->>'height'         AS DOUBLE) AS height,
        TRY_CAST(properties->>'building:levels' AS INT)   AS levels
    FROM read_json(
        'munich_buildings_plain.geojsonl',
        format='newline_delimited',
        columns={geometry: 'JSON', properties: 'JSON'}
    )
""")
con.execute(_height_sql("outlines", include_name_overrides=True))
print(f"  {con.execute('SELECT COUNT(*) FROM outlines').fetchone()[0]:,} outlines loaded")

# 2. Two-pass PBF extraction: building:part ways + building relations
part_features, relation_features = extract_from_pbf("oberbayern.osm.pbf", MUNICH_BBOX)

# 3. Load building:parts into DB
parts_path = _write_temp_geojsonl(part_features)
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
con.execute(_height_sql("parts", include_name_overrides=False))
os.unlink(parts_path)
print(f"  {con.execute('SELECT COUNT(*) FROM parts').fetchone()[0]:,} parts loaded")

# 4. Load building relations into DB
relations_path = _write_temp_geojsonl(relation_features)
con.execute(f"""
    CREATE OR REPLACE TABLE relations AS
    SELECT
        ST_GeomFromGeoJSON(geometry::TEXT)                AS geom,
        properties->>'name'                               AS name,
        properties->>'building'                           AS building_type,
        TRY_CAST(properties->>'height'         AS DOUBLE) AS height,
        TRY_CAST(properties->>'building:levels' AS INT)   AS levels
    FROM read_json(
        '{relations_path}',
        format='newline_delimited',
        columns={{geometry: 'JSON', properties: 'JSON'}}
    )
""")
con.execute(_height_sql("relations", include_name_overrides=True))
os.unlink(relations_path)
print(f"  {con.execute('SELECT COUNT(*) FROM relations').fetchone()[0]:,} relations loaded")

# 5. Find landmark outlines replaced by parts (only for NAME_OVERRIDES buildings)
landmark_names = ", ".join(f"'{n}'" for n in NAME_OVERRIDES)
print("Identifying landmark outlines replaced by parts…")
con.execute(f"""
    CREATE OR REPLACE TABLE landmark_outlines_replaced AS
    SELECT DISTINCT o.rowid
    FROM outlines o
    JOIN parts p ON ST_Intersects(o.geom, p.geom)
    WHERE o.name IN ({landmark_names})
      AND ST_IsValid(o.geom) AND ST_IsValid(p.geom)
      AND ST_Area(ST_Intersection(o.geom, p.geom)) / NULLIF(ST_Area(o.geom), 0) > 0.3
""")
print(f"  {con.execute('SELECT COUNT(*) FROM landmark_outlines_replaced').fetchone()[0]:,} landmark outlines replaced by parts")

con.execute(f"""
    CREATE OR REPLACE TABLE landmark_parts AS
    SELECT DISTINCT p.geom, p.name, p.building_type, p.height, p.levels, p.resolved_height
    FROM parts p
    JOIN outlines o ON ST_Intersects(p.geom, o.geom)
    WHERE o.name IN ({landmark_names})
      AND ST_IsValid(p.geom) AND ST_IsValid(o.geom)
      AND ST_Area(ST_Intersection(p.geom, o.geom)) / NULLIF(ST_Area(p.geom), 0) > 0.3
""")
print(f"  {con.execute('SELECT COUNT(*) FROM landmark_parts').fetchone()[0]:,} parts selected for landmarks")

# 6. Find outlines substantially covered by a relation (>80% overlap).
#    These are typically the outer-ring ways of a relation that were also
#    tagged as standalone buildings in OSM — the relation is more complete.
print("Identifying outlines superseded by relations…")
con.execute("""
    CREATE OR REPLACE TABLE outlines_superseded AS
    SELECT DISTINCT o.rowid
    FROM outlines o
    JOIN relations r ON ST_Intersects(o.geom, r.geom)
    WHERE ST_IsValid(o.geom) AND ST_IsValid(r.geom)
      AND ST_Area(ST_Intersection(o.geom, r.geom)) / NULLIF(ST_Area(o.geom), 0) > 0.8
""")
print(f"  {con.execute('SELECT COUNT(*) FROM outlines_superseded').fetchone()[0]:,} outlines superseded by relations")

# 7. Build final buildings table
print("Building final buildings table…")
con.execute(f"""
    CREATE OR REPLACE TABLE buildings AS

    -- Landmark building:parts (accurate per-part heights for e.g. Frauenkirche)
    SELECT
        CASE WHEN ST_IsEmpty(ST_Simplify(geom, 0.00007))
             THEN geom ELSE ST_Simplify(geom, 0.00007) END AS geom,
        name, building_type, height, levels, resolved_height
    FROM landmark_parts

    UNION ALL

    -- Relation-based buildings (courtyards, multi-wing complexes, etc.)
    -- kept at full resolution since they are often irregular and complex
    SELECT geom, name, building_type, height, levels, resolved_height
    FROM relations
    WHERE resolved_height > 3 OR name IS NOT NULL

    UNION ALL

    -- Way-based outlines: exclude those replaced by parts or superseded by relations
    SELECT
        CASE WHEN ST_IsEmpty(ST_Simplify(geom, 0.00007))
             THEN geom ELSE ST_Simplify(geom, 0.00007) END AS geom,
        name, building_type, height, levels, resolved_height
    FROM outlines
    WHERE rowid NOT IN (SELECT rowid FROM landmark_outlines_replaced)
      AND rowid NOT IN (SELECT rowid FROM outlines_superseded)
      AND (resolved_height > 3 OR name IS NOT NULL)
""")
con.execute("""
    DROP TABLE outlines;
    DROP TABLE parts;
    DROP TABLE relations;
    DROP TABLE landmark_outlines_replaced;
    DROP TABLE landmark_parts;
    DROP TABLE outlines_superseded;
""")
total_buildings = con.execute("SELECT COUNT(*) FROM buildings").fetchone()[0]
print(f"  {total_buildings:,} buildings in final table")

# ── Summary ─────────────────────────────────────────────────────────────────

print("\nHeight resolution summary:")
rows = con.execute("""
    SELECT
        CASE
            WHEN name IN (""" + landmark_names + """)  THEN '0. name override / landmark part'
            WHEN height IS NOT NULL                     THEN '1. explicit height tag'
            WHEN levels IS NOT NULL AND levels > 0      THEN '2. levels × 3m'
            WHEN building_type IN ('garage','garages','shed','barn','farm_auxiliary',
                                   'roof','allotment_house','kiosk','stadium',
                                   'grandstand','bleachers')
                                                        THEN '3. type override'
            ELSE                                             '4. zone default'
        END AS source,
        COUNT(*) AS n
    FROM buildings
    GROUP BY 1 ORDER BY 1
""").fetchall()
total = sum(r[1] for r in rows)
for source, n in rows:
    print(f"  {source}: {n:,} ({n/total*100:.1f}%)")

print("\nSpot checks:")
for name in ['Neues Rathaus', 'Frauenkirche', 'Deutsches Museum', 'BMW-Vierzylinder',
             'Olympiaturm', 'Allianz Arena']:
    rows = con.execute("""
        SELECT name, building_type, resolved_height,
               ST_XMin(geom)+((ST_XMax(geom)-ST_XMin(geom))/2) AS cx,
               ST_YMin(geom)+((ST_YMax(geom)-ST_YMin(geom))/2) AS cy
        FROM buildings WHERE name = ?
        ORDER BY resolved_height DESC LIMIT 1
    """, [name]).fetchall()
    if rows:
        r = rows[0]
        print(f"  {r[0]}: type={r[1]}, resolved={r[2]}m @ ({r[3]:.4f}, {r[4]:.4f})")
    else:
        print(f"  {name}: NOT FOUND")

con.close()

# Compact to eliminate internal fragmentation from intermediate tables
print("\nCompacting database…")
con2 = duckdb.connect("munich_compact.duckdb")
con2.execute("INSTALL spatial; LOAD spatial;")
con2.execute("ATTACH 'munich.duckdb' AS src (READ_ONLY)")
con2.execute("CREATE TABLE buildings AS SELECT * FROM src.buildings")
con2.execute("CHECKPOINT")
con2.close()
shutil.move("munich_compact.duckdb", "munich.duckdb")

final_mb = os.path.getsize("munich.duckdb") / 1024 / 1024
print(f"Saved to munich.duckdb ({final_mb:.0f} MB)")
