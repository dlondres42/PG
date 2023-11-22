import math
from typeguard import check_type

class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __str__(self) -> str:
        return f"Vector ({self.x}, {self.y}, {self.z})" 

    def dot_product(self, other: 'Vector') -> float:
        check_type(other, Vector), "Non Vector value provided" 
        return ((self.x * other.x) + (self.y * other.y) + (self.z * other.z))

    def magnitude(self) -> float:
        return math.sqrt(self.dot_product(self))
    
    def __truediv__(self, other) -> 'Vector':
        if isinstance(other, Vector):
            raise TypeError("Cannot divide a vector by another vector")
        return Vector(self.x/other,self.y/other,self.z/other)

    def normalize(self) -> 'Vector':
        mag = self.magnitude()
        assert mag != 0, "Cannot normalize a zero vector"
        return self/mag
    
    def __add__(self, other: 'Vector') -> 'Vector':
        check_type(other, Vector), "Non Vector value provided"
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        check_type(other, Vector), "Non Vector value provided"
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other) -> 'Vector':
        if isinstance(other, Vector):
            raise TypeError("Cannot multiply a vector by another vector")
        return Vector(self.x*other,self.y*other,self.z*other)
    
    def __rmul__(self, other) -> 'Vector':
        return self.__mul__(other)




#vetor = Vector(1,-2,-2)
#vetor2 = Vector(3,6,9)
#print(vetor2.dot_product(vetor))
#print(vetor / Vector().normalize())