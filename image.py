from color import Color


class Image:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.pixels = [[None for _ in range(width)] for _ in range(height)]

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        self.pixels[y][x] = color

    def write_ppm(self, file):
        file.write(f"P3 {self.width} {self.height}\n255\n")

        def byte(code: float) -> float:
            return round(max(min(code * 255, 255), 0))

        for row in self.pixels:
            for color in row:
                file.write(f"{byte(color.r)} {byte(color.g)} {byte(color.b)} ")
            file.write("\n")
