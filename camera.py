import numpy as np


class Camera:
    def __init__(
        self,
        origin: np.array,
        w: np.array,
        v: np.array,
        u: np.array,
        dist: float,
    ):
        self.origin = origin
        self.w = w
        self.v = v
        self.u = u
        self.dist = dist

    def get_params(self):
        return self.origin, self.w, self.v, self.u, self.dist

    def __str__(self) -> str:
        return f"Camera(origin={self.origin}, w={self.w}, v={self.v}, u={self.u}, dist={self.dist})"

    def __repr__(self) -> str:
        return str(self)
