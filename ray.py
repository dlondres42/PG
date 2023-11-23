from vector import Vector, Point


class Ray:
    """
    Represents a ray in 3D space.

    Attributes:
        origin (Point): The origin point of the ray.
        direction (Vector): The direction vector of the ray.
    """

    def __init__(self, origin: Point, direction: Vector) -> None:
        self.origin = origin
        self.direction = direction.normalize()

    def __str__(self) -> str:
        return f"Ray from point: ({self.origin} to direction: {self.direction})"
