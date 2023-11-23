import math
from typing import Self
from typeguard import check_type


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __str__(self) -> str:
        return f"Vector ({self.x}, {self.y}, {self.z})"

    def dot_product(self, other: "Vector") -> float:
        check_type(other, Vector), "Non Vector value provided"
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    @classmethod
    def cross_product(cls, vector1: "Vector", vector2: "Vector") -> "Vector":
        check_type(vector1, Vector), "Non Vector value provided"
        check_type(vector2, Vector), "Non Vector value provided"
        x = (vector1.y * vector2.z) - (vector1.z * vector2.y)
        y = (vector1.z * vector2.x) - (vector1.x * vector2.z)
        z = (vector1.x * vector2.y) - (vector1.y * vector2.x)
        return cls(x, y, z)

    def magnitude(self) -> float:
        return math.sqrt(self.dot_product(self))

    def __truediv__(self, other) -> "Vector":
        if isinstance(other, Vector):
            raise TypeError("Cannot divide a vector by another vector")
        return Vector(self.x / other, self.y / other, self.z / other)

    def normalize(self) -> "Vector":
        mag = self.magnitude()
        assert mag != 0, "Cannot normalize a zero vector"
        return self / mag

    def __add__(self, other: "Vector") -> "Vector":
        check_type(other, Vector), "Non Vector value provided"
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector") -> "Vector":
        check_type(other, Vector), "Non Vector value provided"
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other) -> "Vector":
        if isinstance(other, Vector):
            raise TypeError("Cannot multiply a vector by another vector")
        return Vector(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other) -> "Vector":
        return self.__mul__(other)


class Point(Vector):
    pass


class Color(Vector):
    """Stores colors as RGB triplets, based of Vector3"""

    @classmethod
    def fromHex(cls, hex="#000000") -> Self.__class__:
        x = int(hex[1:3], 16) / 255.0
        y = int(hex[3:5], 16) / 255.0
        z = int(hex[5:], 16) / 255.0
        return cls(x, y, z)
