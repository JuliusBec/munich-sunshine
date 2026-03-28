import httpx

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

query = """
[out:json];
way["name"="BMW-Vierzylinder"]["building"](around:500,48.1775,11.5563);
out tags;
"""

resp = httpx.post(OVERPASS_URL, data={"data": query}, timeout=30)
elements = resp.json()["elements"]

for el in elements:
    tags = el.get("tags", {})
    name = tags.get("name", "unknown")
    height = tags.get("height") or tags.get("building:height") or tags.get("roof:height")
    print(f"{name}: {height}m")
