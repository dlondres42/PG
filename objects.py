import dis
from vector import Point, Color, Vector
from ray import Ray
from math import sqrt

INF = int(2e9)


class Object3D:
    """
    Represents a 3D object in a scene.
    """

    def __init__(self) -> None:
        pass

    def intersects(self, ray: Ray) -> int:
        pass

    def normal(self, surface_point: Point) -> Vector:
        pass


class Sphere(Object3D):
    def __init__(self, center: Point, radius: float, color: Color) -> None:
        self.center = center
        self.radius = radius
        self.color = color

    def intersection(self, ray: Ray):
        _v = ray.origin - self.center
        a = ray.direction.dot_product(ray.direction)
        b = 2 * _v.dot_product(ray.direction)
        c = _v.dot_product(_v) - (self.radius * self.radius)

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return INF
        elif discriminant == 0:
            return -b / (2 * a)
        else:
            return min(
                (-b + sqrt(discriminant)) / (2 * a), (-b - sqrt(discriminant)) / (2 * a)
            )


class Plane(Object3D):
    """
    Represents a plane in 3D space.

    Attributes:
    - point: A Point object representing any point belonging to the plane.
    - normal: A Vector object representing the direction of the plane's normal.
    - color: A Color object defining the material's color.

    Methods:
    - __init__(point: Point, normal: Vector, color: Color): Initializes a Plane object with the given point, normal, and color.
    - intersects(ray: Ray) -> float | None: Checks if a ray intersects the plane. Returns the distance to the intersection if the ray does intersect, returns None if it does not.
    """

    def __init__(self, point: Point, normal: Vector, color: Color) -> None:
        """
        Initializes a Plane object with the given point, normal, and color.

        Args:
        - point: A Point object representing any point belonging to the plane.
        - normal: A Vector object representing the direction of the plane's normal.
        - color: A Color object defining the material's color.
        """
        self.point = point
        self._normal = normal.normalize()

    def intersects(self, ray: Ray) -> float | None:
        """
        Checks if a ray intersects the plane.

        Args:
        - ray: A Ray object representing the ray to be checked.

        Returns:
        - float | None: The distance to the intersection if the ray does intersect, None if it does not.
        """

        if self._normal.dot_product(ray.direction) >= 0.001:
            distance = (self._normal.dot_product(self.point - ray.origin)) / (
                self._normal.dot_product(ray.direction)
            )
            if distance > 0:
                return distance
