from math import e
from image import Image
from vector import Color, Vector, Point
from camera import Camera
from objects import Sphere, Plane
from scene import Scene
from engine import Engine


def get_input() -> tuple:
    file = open("input.txt", "r")
    lines = file.readlines()
    file.close()

    x, y, z = map(float, lines[0].split()[:3])
    C = Point(x, y, z)
    x, y, z = map(float, lines[1].split()[:3])
    O = Point(x, y, z)  # direção do centro da malha
    w = Vector(O.x - C.x, O.y - C.y, O.z - C.z).normalize()
    x, y, z = map(float, lines[2].split()[:3])
    up = Vector(x, y, z).normalize()  # falso vetor up
    d = int(lines[3].split()[0])
    hres = int(lines[4].split()[0])
    vres = int(lines[5].split()[0])

    return (C, w, up, d, hres, vres)


def main():
    camera_center, w, up, d, hres, vres = get_input()
    WIDTH, HEIGHT = int(hres), int(vres)
    u = Vector.cross_product(w, up).normalize()
    v = Vector.cross_product(u, up).normalize()
    camera = Camera(camera_center, w, u, v, d)
    objects = [
        # Sphere(Point(0, 0, 3), 3, Color(0, 255, 0)),
        Sphere(Point(-0.93, -20.05, 30), 2, Color(255, 0, 0)),
        # Plane(Point(0, -10, 0), Vector(0, 10, 0), Color(0, 0, 255)),
    ]
    scene = Scene(camera, objects, WIDTH, HEIGHT)
    engine = Engine()
    image = engine.render(scene)

    with open("output.ppm", "w") as img_file:
        image.write_ppm(img_file)


if __name__ == "__main__":
    main()
