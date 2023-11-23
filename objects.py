import numpy as np
from math import sqrt

INF = float(2e9 + 7)


class Object:
    def __init__(self, center: np.array, color: tuple):
        self.center = center
        self.color = color

    def __str__(self) -> str:
        return f"Object(center={self.center}, radius={self.radius}, color={self.color})"

    def __repr__(self) -> str:
        return str(self)

    def intersect(self, *args) -> float:
        pass


class Sphere(Object):
    def __init__(self, center: np.array, radius: float, color: tuple):
        super().__init__(center, color)
        self.radius = radius

    def intersect(self, camera_center, d_vector) -> float:
        co = camera_center - self.center
        a = np.dot(d_vector, d_vector)
        b = 2 * np.dot(d_vector, co)
        c = np.dot(co, co) - self.radius**2
        determinant = b**2 - 4 * a * c
        if determinant < 0:
            return INF
        t1 = (-b + sqrt(determinant)) / (2 * a)
        t2 = (-b - sqrt(determinant)) / (2 * a)
        return min(t1, t2)
