import numpy as np
from camera import Camera
from color import Color
from objects import Sphere, Plane, Triangles
from scene import Scene
from PIL import Image

INF = float(2e9 + 7)


def main():
    # inputs do usuario
    O = np.array([0, 0, 0])  # origem
    A = np.array([2, 0, 0])# alvo 
    up = np.array([0, 1, 0])  # vetor up 
    dist = 1 # distancia do alvo
    hres = vres = 500  # resolucao horizontal e vertical

    # calculo dos vetores
    w = normalize(A - O)
    u = normalize(np.cross(up, w)) 
    v = normalize(np.cross(w, u)) * -1

    # objetos
    test = Triangles(2, 4, np.array([[2, -0.3, 0], [2.5,0.1,0.2], [1.7, 0.5, -0.1], [1.5,0.3,1.2]]), [(0,1,2), (0,2,3)], Color(0, 255, 0))
    camera = Camera(O, w, u, v, dist)
    objects = [
        #Sphere(np.array([2, -0.3, 0]), 0.5, Color(0, 255, 0)),
        Plane(np.array([0,-1,0]), np.array([0,1,0]), Color(0,0,255)),
        Sphere(np.array([4,1,1]), 0.3, Color(155,133,200)),
        Triangles(2, 4, np.array([[4, 1, 0], [4,1,1], [4, 1, -1], [4,0,0]]), [(0,1,3), (1,2,3)], Color(133, 107, 55))
    ]
    scene = Scene(camera, objects, hres, vres)
    mtx = render(scene)
    image = Image.fromarray(mtx)
    # image.save("output.png")  # save img
    image.show()  # show img


def render(scene: Scene) -> np.array:
    hres, vres = scene.width, scene.height
    camera = scene.camera
    C, w, u, v, dist = camera.get_params()
    # base ortornormal w, u, v

    # malha
    tam_x, tam_y = 0.5, 0.5
    desl_h = ((2 * tam_x) / (hres - 1)) * u
    desl_v = ((2 * tam_y) / (vres - 1)) * v
    vet_inicial = (w * dist) - (tam_x * u) - (tam_y * v)

    mtx = np.zeros((scene.height, scene.width, 3), dtype=np.uint8)

    for j in range(vres):  # iterando sobre as linhas
        for i in range(hres):  # iterando sobre as colunas
            v_r = (
                vet_inicial + (i * desl_h) + (j * desl_v)
            )  # ponto do centro de cada pixel
            _, color = ray_color(C, v_r, scene)
            mtx[j][i] = color.to_list()

    return mtx


def find_nearest(ray_origin, ray_direction, scene: Scene):
    t_min = INF
    obj_hit = None
    for obj in scene.objects:
        t = obj.intersect(ray_origin, ray_direction)
        if t < t_min and t > 0.01:
            t_min = t
            obj_hit = obj
    return t_min, obj_hit


def ray_color(ray_origin, ray_direction, scene: Scene):
    t_min, obj_hit = find_nearest(ray_origin, ray_direction, scene)
    color = Color(229, 255, 204)
    if obj_hit is not None:
        color = obj_hit.color
    return t_min, color


def normalize(vect: np.array):
    norm = np.linalg.norm(vect)
    if norm == 0:
        return vect
    return vect / norm


main()
