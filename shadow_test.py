import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import translate
from datetime import datetime, timezone

# --- Sun position at 1pm Munich time (CET = UTC+1) ---
dt_utc = datetime(2026, 3, 26, 12, 0, 0, tzinfo=timezone.utc)  # 13:00 CET
lat, lon = 48.1771, 11.5601  # BMW Tower centre

times = pd.DatetimeIndex([dt_utc])
solar = pvlib.solarposition.get_solarposition(times, lat, lon)
azimuth = float(solar["azimuth"].iloc[0])
elevation = float(solar["apparent_elevation"].iloc[0])
print(f"Sun azimuth: {azimuth:.1f}°, elevation: {elevation:.1f}°")

# --- BMW-Vierzylinder footprint (OSM way 146381661, height=100m) ---
height = 100.0
geometry = [
    {"lat": 48.1768825, "lon": 11.560345}, {"lat": 48.1769255, "lon": 11.5603463},
    {"lat": 48.176928, "lon": 11.5603566}, {"lat": 48.1769346, "lon": 11.5603759},
    {"lat": 48.1769505, "lon": 11.5604059}, {"lat": 48.1769694, "lon": 11.5604274},
    {"lat": 48.1769778, "lon": 11.5604328}, {"lat": 48.1769912, "lon": 11.5604413},
    {"lat": 48.177021, "lon": 11.5604471}, {"lat": 48.1770413, "lon": 11.5604431},
    {"lat": 48.1770548, "lon": 11.560436}, {"lat": 48.1770599, "lon": 11.5604334},
    {"lat": 48.1770683, "lon": 11.5604249}, {"lat": 48.1770859, "lon": 11.5604069},
    {"lat": 48.177109, "lon": 11.5603565}, {"lat": 48.177116, "lon": 11.5603148},
    {"lat": 48.1771129, "lon": 11.5602603}, {"lat": 48.1770988, "lon": 11.5602166},
    {"lat": 48.1770981, "lon": 11.5602144}, {"lat": 48.177078, "lon": 11.5601834},
    {"lat": 48.17707, "lon": 11.5601766}, {"lat": 48.177055, "lon": 11.5601638},
    {"lat": 48.1770553, "lon": 11.5601367}, {"lat": 48.1770557, "lon": 11.5601035},
    {"lat": 48.1770567, "lon": 11.5601029}, {"lat": 48.1770673, "lon": 11.5600964},
    {"lat": 48.1770706, "lon": 11.5600935}, {"lat": 48.1770798, "lon": 11.5600855},
    {"lat": 48.1770988, "lon": 11.5600599}, {"lat": 48.1771119, "lon": 11.5600303},
    {"lat": 48.1771199, "lon": 11.5599971}, {"lat": 48.1771221, "lon": 11.5599522},
    {"lat": 48.1771183, "lon": 11.5599221}, {"lat": 48.1771109, "lon": 11.5598949},
    {"lat": 48.1770918, "lon": 11.5598573}, {"lat": 48.1770724, "lon": 11.5598394},
    {"lat": 48.1770706, "lon": 11.5598378}, {"lat": 48.1770571, "lon": 11.5598254},
    {"lat": 48.177029, "lon": 11.559817}, {"lat": 48.1769927, "lon": 11.5598246},
    {"lat": 48.1769819, "lon": 11.5598335}, {"lat": 48.1769628, "lon": 11.5598492},
    {"lat": 48.1769431, "lon": 11.5598811}, {"lat": 48.1769331, "lon": 11.5599092},
    {"lat": 48.1768982, "lon": 11.5599069}, {"lat": 48.1768894, "lon": 11.5598821},
    {"lat": 48.1768726, "lon": 11.5598532}, {"lat": 48.1768532, "lon": 11.5598331},
    {"lat": 48.1768418, "lon": 11.5598266}, {"lat": 48.1768312, "lon": 11.5598205},
    {"lat": 48.1768013, "lon": 11.5598165}, {"lat": 48.1767813, "lon": 11.5598216},
    {"lat": 48.1767781, "lon": 11.5598234}, {"lat": 48.176763, "lon": 11.5598322},
    {"lat": 48.1767376, "lon": 11.55986}, {"lat": 48.1767312, "lon": 11.559875},
    {"lat": 48.1767236, "lon": 11.5598928}, {"lat": 48.1767157, "lon": 11.5599114},
    {"lat": 48.176714, "lon": 11.5599229}, {"lat": 48.1767096, "lon": 11.5599534},
    {"lat": 48.1767137, "lon": 11.5600038}, {"lat": 48.176714, "lon": 11.560008},
    {"lat": 48.1767298, "lon": 11.5600534}, {"lat": 48.1767507, "lon": 11.5600835},
    {"lat": 48.1767644, "lon": 11.5600957}, {"lat": 48.1767628, "lon": 11.5601499},
    {"lat": 48.1767492, "lon": 11.5601577}, {"lat": 48.1767274, "lon": 11.5601785},
    {"lat": 48.1767127, "lon": 11.5602023}, {"lat": 48.1767112, "lon": 11.5602048},
    {"lat": 48.1766997, "lon": 11.560236}, {"lat": 48.1766928, "lon": 11.5602801},
    {"lat": 48.1766934, "lon": 11.5603108}, {"lat": 48.176698, "lon": 11.5603395},
    {"lat": 48.1767119, "lon": 11.5603782}, {"lat": 48.1767129, "lon": 11.560381},
    {"lat": 48.1767437, "lon": 11.5604206}, {"lat": 48.1767481, "lon": 11.560423},
    {"lat": 48.1767704, "lon": 11.5604355}, {"lat": 48.1768068, "lon": 11.5604368},
    {"lat": 48.1768268, "lon": 11.5604264}, {"lat": 48.1768388, "lon": 11.5604201},
    {"lat": 48.1768616, "lon": 11.5603938}, {"lat": 48.1768742, "lon": 11.560369},
    {"lat": 48.1768825, "lon": 11.560345},
]
coords = [(n["lon"], n["lat"]) for n in geometry]
footprint = Polygon(coords)

# --- Shadow polygon ---
m_per_deg_lat = 111_000
m_per_deg_lon = 111_000 * np.cos(np.radians(lat))

shadow_length = height / np.tan(np.radians(elevation))
d_lon = -shadow_length * np.sin(np.radians(azimuth)) / m_per_deg_lon
d_lat = -shadow_length * np.cos(np.radians(azimuth)) / m_per_deg_lat

shadow = footprint.union(translate(footprint, xoff=d_lon, yoff=d_lat)).convex_hull

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 8))

sx, sy = shadow.exterior.xy
ax.fill(sx, sy, color="steelblue", alpha=0.4, label=f"Shadow (~{shadow_length:.0f}m long)")

bx, by = footprint.exterior.xy
ax.fill(bx, by, color="dimgray", label="BMW-Vierzylinder")

ax.set_aspect("equal")
ax.set_title(f"BMW-Vierzylinder shadow at 1pm CET\nSun: azimuth {azimuth:.1f}°, elevation {elevation:.1f}°")
ax.legend()
plt.tight_layout()
plt.savefig("shadow.png", dpi=150)
print("Saved shadow.png")
