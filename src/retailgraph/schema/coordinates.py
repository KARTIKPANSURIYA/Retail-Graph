"""Explicit coordinate conversions."""

from retailgraph.schema.records import NormalizedPoint


def pixel_to_normalized(x: float, y: float, width: int, height: int) -> NormalizedPoint:
    """Convert pixel-center coordinates using x/width and y/height."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not (0 <= x <= width and 0 <= y <= height):
        raise ValueError("pixel coordinate lies outside image bounds")
    return NormalizedPoint(x=x / width, y=y / height)


def normalized_to_pixel(point: NormalizedPoint, width: int, height: int) -> tuple[float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return point.x * width, point.y * height
