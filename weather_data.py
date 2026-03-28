import pvlib
import pandas as pd
from datetime import datetime, timezone
import httpx

lat, lon = 48.1351, 11.5820  # Munich

now = datetime.now(timezone.utc)
times = pd.DatetimeIndex([now])

solar_position = pvlib.solarposition.get_solarposition(times, lat, lon)
print(solar_position[['azimuth', 'apparent_elevation']])

resp = httpx.get("https://api.open-meteo.com/v1/forecast", params={
    "latitude": lat,
        "longitude": lon,
    "current": "cloud_cover",
})

data = resp.json()
print("cloud cover:", data["current"]["cloud_cover"])  # 0-100%