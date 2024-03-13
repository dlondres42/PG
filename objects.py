import numpy as np
from math import sqrt

INF = float(2e9 + 7)


class Material:
    def __init__(
        self,
        kd=np.array([0.25, 0.25, 0.25]),
        ks=np.array([0.25, 0.25, 0.25]),
        ka=np.array([0.25, 0.25, 0.25]),
        kr=np.array([0.25, 0.25, 0.25]),
        kt=np.array([0.25, 0.25, 0.25]),
        eta=5,
        ior=1,
    ):
        self.kd = kd  # Diffuse coefficient
        self.ks = ks  # Specular coefficient
        self.ka = ka  # Ambient coefficient
        self.kr = kr  # Reflection coefficient
        self.kt = kt  # Transmission coefficient
        self.eta = eta  # Roughness coefficient
        self.ior = ior  # Index of refraction


class Object:
    def __init__(self, color: tuple, material: Material = Material()):
        self.color = color
        self.material = material

    def __str__(self) -> str:
        return f"Object(color={self.color}, material={self.material})"

    def __str__(self) -> str:
        return f"Object(color={self.color}, kd={self.kd}, ks={self.ks}, ka={self.ka}, kr={self.kr}, kt={self.kt}, eta={self.eta})"

    def __repr__(self) -> str:
        return str(self)

    def intersect(self, *args) -> float:
        pass


class Sphere(Object):
    def __init__(
        self,
        center: np.array,
        radius: float,
        color: tuple,
        material: Material = Material(),
    ):
        super().__init__(color, material)
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
    def __init__(
        self,
        center: np.array,
        normal: np.array,
        color: tuple,
        material: Material = Material(),
    ):
        super().__init__(color, material)
        self.normal = normal
        self.center = center

    def intersect(self, camera_center, d_vector) -> float:
        a = np.dot(self.normal, d_vector)
        if a == 0:
            return INF
        return (
            np.dot(self.normal, self.center) - np.dot(self.normal, camera_center)
        ) / a


class Triangle:
    def __init__(self, vertices: np.array) -> None:
        # self.point1, self.point2, self.point3 = self.ensure_counterclockwise(vertices)
        self.point1 = vertices[0]
        self.point2 = vertices[1]
        self.point3 = vertices[2]
        self.normal = (
            np.cross(self.point1 - self.point2, self.point1 - self.point3) * -1
        )

    @staticmethod
    def ensure_counterclockwise(vertices: np.array) -> np.array:
        v0, v1, v2 = vertices
        if np.dot(np.cross(v1 - v0, v2 - v0), v0) < 0:
            return v0, v2, v1
        return vertices

    def calculate_normal(self) -> np.array:
        vertices = self.ensure_counterclockwise([self.point1, self.point2, self.point3])
        v0, v1, v2 = vertices
        return np.cross(v1 - v0, v2 - v0)

    def __str__(self) -> str:
        return f"ponto1 = {self.point1} \n ponto2 = {self.point2} \n ponto3 = {self.point3} \n normal = {self.normal}"


class Triangles(Object):
    def __init__(
        self,
        num_triangles: int,
        num_vertices: int,
        vertices: np.array,
        triangle_index: np.array,
        color: tuple,
        material: Material = Material(),
    ):
        super().__init__(color, material)
        self.num_triangles = num_triangles
        self.num_vertices = num_vertices
        self.vertices = vertices
        self.triangle_index = triangle_index
        self.triangles = []
        self.define_triangles()

    def define_triangles(self) -> None:
        for i in range(self.num_triangles):
            indexes = self.triangle_index[i]
            triangle = Triangle(
                [
                    self.vertices[indexes[0]],
                    self.vertices[indexes[1]],
                    self.vertices[indexes[2]],
                ]
            )
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
            # w = camera_center - v0
            # (np.dot(self.normal, self.center) - np.dot(self.normal, camera_center))
            t = (np.dot(normal, v0) - np.dot(normal, camera_center)) / ndotu
            intersection_point = camera_center + t * d_vector

            # Check if the intersection point is inside the triangle using barycentric coordinates
            w0 = v1 - v0
            w1 = v2 - v0
            w2 = intersection_point - v0

            u = (np.dot(w1, w1) * np.dot(w2, w0) - np.dot(w0, w1) * np.dot(w2, w1)) / (
                np.dot(w0, w0) * np.dot(w1, w1) - np.dot(w0, w1) ** 2
            )

            v = (np.dot(w0, w0) * np.dot(w2, w1) - np.dot(w0, w1) * np.dot(w2, w0)) / (
                np.dot(w0, w0) * np.dot(w1, w1) - np.dot(w0, w1) ** 2
            )

            if 0 <= u <= 1 and 0 <= v <= 1 and u + v <= 1:
                min_t = min(min_t, t)

        return min_t if min_t != INF else INF

    def normal_at(self, intersection_point):
        # Find the triangle closest to the intersection point
        closest_triangle = None
        min_distance = INF
        for triangle in self.triangles:
            for vertex in [triangle.point1, triangle.point2, triangle.point3]:
                distance = np.linalg.norm(vertex - intersection_point)
                if distance < min_distance:
                    closest_triangle = triangle
                    min_distance = distance
        return closest_triangle.normal


class BezierSurface(Object):
    def __init__(self, control_points: np.array, color: tuple, material: Material = Material()):
        super().__init__(color, material)
        self.u_degree = len(control_points) - 1
        self.v_degree = len(control_points[0]) - 1
        self.control_points = control_points
        self.triangles = self.triangulate()

    def __str__(self) -> str:
        return f"BezierSurface(color={self.color}, (u,v)= ({self.u}, {self.v}), material={self.material}, control_points={self.control_points},)"
    
    def evaluate(self, u, v):
        """
        Evaluate the position on the Bezier surface at parameters (u, v).
        
        Parameters:
            u (float): Parameter along the u-direction, typically in [0, 1].
            v (float): Parameter along the v-direction, typically in [0, 1].
            
        Returns:
            np.array: The position (x, y, z) on the surface.
        """
        n = self.u_degree
        m = self.v_degree
        result = np.zeros(3)

        for i in range(n + 1):
            for j in range(m + 1):
                coeff = self.bernstein(n, i, u) * self.bernstein(m, j, v)
                result += coeff * self.control_points[i][j]  

        return result

    def bernstein(self, n, i, t):
        """
        Bernstein polynomial value for given parameters.
        
        Parameters:
            n (int): Degree of the polynomial.
            i (int): Index of the basis function.
            t (float): Parameter value typically in [0, 1].
        
        Returns:
            float: Value of the Bernstein polynomial.
        """
        return np.math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    def triangulate(self, num_samples=30):
        # Parametrization
        points = np.zeros((num_samples, num_samples, 3))
        for i in range(num_samples):
            for j in range(num_samples):
                u = i / (num_samples - 1)
                v = j / (num_samples - 1)
                points[i, j] = self.evaluate(u, v)

        flat_points = points.reshape((-1, 3))

        # Triangulation
        triangle_indices = []
        for i in range(num_samples - 1):
            for j in range(num_samples - 1):

                p1_idx = i * num_samples + j
                p2_idx = i * num_samples + (j + 1)
                p3_idx = (i + 1) * num_samples + j
                p4_idx = (i + 1) * num_samples + (j + 1)

                triangle_indices.append([p1_idx, p2_idx, p3_idx])
                triangle_indices.append([p2_idx, p4_idx, p3_idx])

        # Mesh creation
        mesh = Triangles(num_triangles=len(triangle_indices),
                                   num_vertices=len(flat_points),
                                   vertices=flat_points,
                                   triangle_index=triangle_indices,
                                   color=self.color)
        return mesh

    # Both intersect and normal finding functions are equivalent to the meshes'
    def intersect(self, camera_center, d_vector) -> float:
        return self.triangles.intersect(camera_center, d_vector)
    
    def normal_at(self, intersection_point):
        return self.triangles.normal_at(intersection_point)