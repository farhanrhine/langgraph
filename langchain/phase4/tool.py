from langchain.tools import tool
from typing import Optional

# 1. Define the function and use the @tool decorator
@tool
def get_current_weather(city: str) -> str:
    """Returns the current weather in a given city."""
    if "tokyo" in city.lower():
        return "The current temperature in Tokyo is 15°C and sunny."
    elif "delhi" in city.lower():
        return "It is currently 32°C and hazy in Delhi."
    else:
        return f"The weather for {city} is not available."

# 2. The Tool object is automatically created
print(f"Tool Description: {get_current_weather.description}")

# 3. Example of calling the tool directly (the agent does this)
print(get_current_weather.run("dubai"))