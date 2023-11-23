from vector import Vector, Point, Color


class Camera:
    """
    Represents a camera in a 3D scene.

    Attributes:
        C (Point): The position of the center of the camera.
        w (Vector): The direction the camera is facing.
        u (Vector): The horizontal direction of the camera.
        v (Vector): The vertical direction of the camera.
        d (float): The distance between the camera and the image plane.
    """

    def __init__(self, C: Point, w: Vector, u: Vector, v: Vector, d: float):
        self.w = w
        self.u = u
        self.v = v
        self.d = d
        self.C = C

    def __str__(self):
        return f"Camera at point: {self.C} with w: {self.w}, u: {self.u}, v: {self.v} and d: {self.d}"

    def params(self):
        return self.C, self.w, self.u, self.v, self.d
