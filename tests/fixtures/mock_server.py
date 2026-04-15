"""Minimal FastMCP server for integration tests.

Run as a stdio MCP server:
    python tests/fixtures/mock_server.py

Or reference in mcp.json:
    {
        "name": "test",
        "transport": "stdio",
        "command": "python",
        "args": ["tests/fixtures/mock_server.py"]
    }
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-server")


@mcp.tool()
def get_greeting(name: str) -> str:
    """Get a friendly greeting for a person by name."""
    return f"Hello, {name}! Welcome to umcp."


@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two integers together and return the sum."""
    return a + b


@mcp.tool()
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get mock weather data for a city.

    Args:
        city: The city name.
        unit: Temperature unit — 'celsius' or 'fahrenheit'.
    """
    temps = {"tokyo": 22, "london": 15, "new york": 18, "sydney": 25}
    temp = temps.get(city.lower(), 20)
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32)
    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": "Sunny",
    }


@mcp.tool()
def list_items(category: str) -> list:
    """List mock items in a category."""
    catalog = {
        "fruits": ["apple", "banana", "cherry"],
        "colors": ["red", "green", "blue"],
    }
    return catalog.get(category.lower(), [])


if __name__ == "__main__":
    mcp.run()
