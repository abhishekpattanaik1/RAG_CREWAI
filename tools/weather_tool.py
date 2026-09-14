import os
import requests
from crewai.tools import BaseTool

class WeatherTool(BaseTool):
    name: str = "Weather Tool"
    description: str = (
        "Fetches current weather data for a given city. "
        "Input should be a city name, e.g. 'Hyderabad' or 'London,UK'."
    )

    def _run(self, city: str) -> str:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return f"Could not fetch weather for {city}: {resp.json().get('message', 'unknown error')}"
        data = resp.json()
        return (
            f"Weather in {city}: {data['weather'][0]['description']}, "
            f"Temp: {data['main']['temp']}°C (feels like {data['main']['feels_like']}°C), "
            f"Humidity: {data['main']['humidity']}%, "
            f"Wind: {data['wind']['speed']} m/s"
        )
