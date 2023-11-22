from vector import Vector

# Essa classe serve como um alias para a classe vector.
class Color(Vector):
# Esse class method permite que possamos trabalhar com cores em
# formato hexadecimal. Facilitando a definição da cor. 
    @classmethod
    def convert_hex(cls, hexcolor="#000000") -> 'Color':
        x = int(hexcolor[1:3], 16)/255.0
        y = int(hexcolor[3:5], 16)/255.0
        z = int(hexcolor[5:], 16)/255.0
        return cls(x,y,z)
