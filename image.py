from color import Color
from typeguard import check_type

class Image:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.pixels = [[None for _ in range(width)] for _ in range(height)]

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        #check_type(color, Color), "Invalid color type provided"
        self.pixels[y][x] = color

    def write_ppm(self, file) -> None:
        file.write(f"P3 {self.width} {self.height}\n255\n")

        def byte(code: float) -> float:
            return round(max(min(code * 255, 255), 0))
        
        for row in self.pixels:
            for color in row:
                file.write(f"{byte(color.x)} {byte(color.y)} {byte(color.z)} ")
            file.write("\n")

