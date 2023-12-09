import numpy as np
from math import sqrt

INF = float(2e9 + 7)


class Object:
    def __init__(self, color: tuple):
        self.color = color

    def __str__(self) -> str:
        return f"Object(center={self.center}, radius={self.radius}, color={self.color})"

    def __repr__(self) -> str:
        return str(self)

    def intersect(self, *args) -> float:
        pass


class Sphere(Object):
    def __init__(self, center: np.array, radius: float, color: tuple):
        super().__init__(color)
        self.radius = radius
        self.center = center

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

class Plane(Object):
    def __init__(self, center: np.array, normal: np.array, color: tuple):
        super().__init__(color)
        self.normal = normal
        self.center = center

    def intersect(self, camera_center, d_vector) -> float:
        a = np.dot(self.normal, d_vector)
        if a == 0:
            return INF
        return (np.dot(self.normal, self.center) - np.dot(self.normal, camera_center))/a
        

class Triangle():
    def __init__(self, vertices: np.array) -> None:
        #self.point1, self.point2, self.point3 = self.ensure_counterclockwise(vertices)
        self.point1 = vertices[0]
        self.point2 = vertices[1]
        self.point3 = vertices[2]
        self.normal = np.cross(self.point1 - self.point2, self.point1 - self.point3)
    

    @staticmethod
    def ensure_counterclockwise(vertices: np.array) -> np.array:
        if np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])[2] < 0:
            return vertices[::-1]
        return vertices


    def __str__(self) -> str:
        return f"ponto1 = {self.point1} \n ponto2 = {self.point2} \n ponto3 = {self.point3} \n normal = {self.normal}"

class Triangles(Object):
    def __init__(self, num_triangles: int, num_vertices: int, vertices: np.array, triangle_index: np.array, color: tuple):
        super().__init__(color)
        self.num_triangles = num_triangles
        self.num_vertices = num_vertices
        self.vertices = vertices
        self.triangle_index = triangle_index
        self.triangles = []
        self.define_triangles()

    def define_triangles(self) -> None:
        for i in range(self.num_triangles):
            indexes = self.triangle_index[i]
            triangle = Triangle([self.vertices[indexes[0]], self.vertices[indexes[1]], self.vertices[indexes[2]]])
            self.triangles.append(triangle)


    def intersect(self, camera_center, d_vector) -> float:
        min_t = INF
        for triangle in self.triangles:
            
            v0, v1, v2 = triangle.point1, triangle.point2, triangle.point3

            
            normal = triangle.normal

           
            ndotu = np.dot(normal, d_vector)
            if np.abs(ndotu) < 1e-6:
                continue  

            # Compute the intersection point with the plane of the triangle
            #w = camera_center - v0
            #(np.dot(self.normal, self.center) - np.dot(self.normal, camera_center))
            t = (np.dot(normal, v0)  - np.dot(normal, camera_center))/ ndotu
            intersection_point = camera_center + t * d_vector

            # Check if the intersection point is inside the triangle using barycentric coordinates
            w0 = v1 - v0
            w1 = v2 - v0
            w2 = intersection_point - v0

            u = (np.dot(w1, w1) * np.dot(w2, w0) - np.dot(w0, w1) * np.dot(w2, w1)) / \
                (np.dot(w0, w0) * np.dot(w1, w1) - np.dot(w0, w1) ** 2)

            v = (np.dot(w0, w0) * np.dot(w2, w1) - np.dot(w0, w1) * np.dot(w2, w0)) / \
                (np.dot(w0, w0) * np.dot(w1, w1) - np.dot(w0, w1) ** 2)

            if 0 <= u <= 1 and 0 <= v <= 1 and u + v <= 1:
                min_t = min(min_t, t)

        return min_t if min_t != INF else INF