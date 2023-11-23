from vector import Point
from color import Color
from ray import Ray
from math import sqrt

class Sphere:
    def __init__(self, center: Point, radius: float, color: Color) -> None:
        self.center = center
        self.radius = radius
        self.color = color

    def intersection(self, ray: Ray):

        sphere_to_ray = ray.origin - self.center

        # a = 1
        b = 2 * ray.direction.dot_product(sphere_to_ray)
        c = sphere_to_ray.dot_product(sphere_to_ray) - (self.radius ** 2)
        delta = (b**2) - (4*c)

        if delta >= 0:
            dist = (-b - sqrt(delta)/ 2)
            if dist > 0:
                return dist
        return None
        
