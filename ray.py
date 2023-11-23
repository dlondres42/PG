from vector import Vector, Point

class Ray:
    def __init__(self, origin: Point, direction: Vector) -> None:
        self.origin = origin
        self.direction = direction.normalize()
