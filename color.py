class Color:
    # Esse class method permite que possamos trabalhar com cores em
    # formato hexadecimal. Facilitando a definição da cor.
    def __init__(self, r: int, g: int, b: int) -> None:
        self.r = r % 256
        self.g = g % 256
        self.b = b % 256

    def __str__(self) -> str:
        return f"Red: {self.r}, Green: {self.g}, Blue: {self.b}"

    def __repr__(self) -> str:
        return str(self)

    def to_list(self) -> list:
        return [self.r, self.g, self.b]
