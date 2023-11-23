from vector import Color, Point


class Light:
    """
    Represents a light source in a 3D scene.

    Attributes:
        position (Point): The position of the light source.
        color (Color): The color of the light source.
    """

    def __init__(self, position: Point, color: Color.fromHex("#FFFFFF")) -> None:
        self.position = position
        self.color = color

    def __str__(self) -> str:
        return f"Light at point: {self.position} with color: {self.color}"
