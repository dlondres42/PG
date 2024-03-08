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
        self.control_points = control_points

    def __str__(self) -> str:
        return f"BezierSurface(color={self.color}, material={self.material}, control_points={self.control_points})"

    def __repr__(self) -> str:
        return str(self)
    
    def evaluate(self, u, v):
        # Evaluate the Bezier surface at parameter values (u, v)
        # Compute the blending functions for each direction (u and v)
        blend_u = self._blend_function(u)
        blend_v = self._blend_function(v)

        # Compute the surface point using the tensor product of the blending functions
        surface_point = np.zeros(3)
        for i in range(len(self.control_points)):
            for j in range(len(self.control_points[0])):
                surface_point += blend_u[i] * blend_v[j] * self.control_points[i][j]

        return surface_point

    def _blend_function(self, t):
        # Compute the blending function for parameter t
        n = len(self.control_points) - 1  # Degree of the Bezier surface
        blend = np.zeros(n + 1)
        for i in range(n + 1):
            blend[i] = self._binomial_coefficient(n, i) * (1 - t)**(n - i) * t**i
        return blend

    def _binomial_coefficient(self, n, k):
        # Compute the binomial coefficient C(n, k)
        return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))
    
    def _intersect_patch(self, ray_origin, ray_direction, p00, p01, p10, p11):
        # Compute vertices of triangles formed by the patch
        vertices = [
            p00, p01, p11,
            p00, p11, p10
        ]

        # Define a Triangles object
        patch_triangles = Triangles(2, len(vertices), np.array(vertices), [(0, 1, 2), (0, 2, 3)], self.color, self.material)

        # Compute intersection using Triangles' intersect method
        t = patch_triangles.intersect(ray_origin, ray_direction)

        return t
    

    def intersect(self, ray_origin, ray_direction):
        min_t = INF
        # Iterate over the control points to form the Bezier patches
        for i in range(len(self.control_points) - 1):
            for j in range(len(self.control_points[0]) - 1):
                # Extract control points for the current patch
                p00 = self.control_points[i][j]
                p01 = self.control_points[i][j+1]
                p10 = self.control_points[i+1][j]
                p11 = self.control_points[i+1][j+1]
                # Compute intersection with the patch
                t = self._intersect_patch(ray_origin, ray_direction, p00, p01, p10, p11)
                # Update minimum intersection distance
                min_t = min(min_t, t)
        return min_t if min_t != INF else INF
    
    def calculate_bezier_normal(self, u, v):
        # Adjusted computation of partial derivatives
        blend_u = self._blend_function(u)
        blend_v = self._blend_function(v)

        dP_du = np.zeros(3)
        dP_dv = np.zeros(3)

        # Compute the partial derivatives of the surface point with respect to u and v
        n_u = len(self.control_points) - 1  # u direction control points count
        n_v = len(self.control_points[0]) - 1  # v direction control points count

        for i in range(n_u):
            for j in range(n_v):
                if i < n_u:  # Ensure within bounds for u direction
                    dP_du += (blend_u[i + 1] - blend_u[i]) * blend_v[j] * self.control_points[i][j]
                if j < n_v:  # Ensure within bounds for v direction
                    dP_dv += blend_u[i] * (blend_v[j + 1] - blend_v[j]) * self.control_points[i][j]

        # Compute the normal vector by taking the cross product of the partial derivatives
        normal = np.cross(dP_du, dP_dv)

        return normal / np.linalg.norm(normal)
    
    def evaluate_derivative_u(self, u, v, epsilon=1e-6):
        # Evaluate surface points at u and u + epsilon
        P_u = self.evaluate(u, v)
        P_u_epsilon = self.evaluate(u + epsilon, v)

        # Approximate derivative with finite differences
        dP_du = (P_u_epsilon - P_u) / epsilon

        return dP_du

    def evaluate_derivative_v(self, u, v, epsilon=1e-6):
        # Evaluate surface points at v and v + epsilon
        P_v = self.evaluate(u, v)
        P_v_epsilon = self.evaluate(u, v + epsilon)

        # Approximate derivative with finite differences
        dP_dv = (P_v_epsilon - P_v) / epsilon

        return dP_dv


    def compute_uv_for_intersection(self, intersection_point, learning_rate=0.01, max_iterations=100, epsilon=1e-6):
        # Initial guess for parameters (u, v)
        u, v = 0.5, 0.5

        for _ in range(max_iterations):
            # Evaluate surface point at current (u, v)
            P = self.evaluate(u, v)

            # Compute the gradient of the Euclidean distance squared between P and intersection_point
            dP_du = self.evaluate_derivative_u(u, v)
            dP_dv = self.evaluate_derivative_v(u, v)

            grad_distance_u = np.dot(dP_du, P - intersection_point)
            grad_distance_v = np.dot(dP_dv, P - intersection_point)

            # Update (u, v) using gradient descent
            u -= learning_rate * grad_distance_u
            v -= learning_rate * grad_distance_v

            # Check for convergence
            if np.sqrt(grad_distance_u**2 + grad_distance_v**2) < epsilon:
                break

        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        return u, v