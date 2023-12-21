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
        affine_transform(Plane(np.array([1,-0.5,0]), np.array([0,1,0]), Color(0,0,255)), translation=(0,0,0), rotation_angles=(0,0,0)),
        affine_transform(Sphere(np.array([4,1,1]), 0.3, Color(155,133,200)), translation=(2,1,0), rotation_angles=(0,0,0)),
        affine_transform(Triangles(2, 4, np.array([[4, 1, 0], [4,1,1], [4, 1, -1], [4,0,0]]), [(0,1,3), (1,2,3)], Color(250, 70, 55)), translation=(0,0,0), rotation_angles=(60,0,0))
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

def affine_transform(obj, translation=(0, 0, 0), rotation_angles=(0, 0, 0)):
    rotation_angles = np.radians(rotation_angles)

    rotation_matrix_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0]), 0],
        [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0]), 0],
        [0, 0, 0, 1]
    ])

    rotation_matrix_y = np.array([
        [np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1]), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1]), 0],
        [0, 0, 0, 1]
    ])

    rotation_matrix_z = np.array([
        [np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0, 0],
        [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    translation_matrix = np.array([
        [1, 0, 0, translation[0]],
        [0, 1, 0, translation[1]],
        [0, 0, 1, translation[2]],
        [0, 0, 0, 1]
    ])

    # Combine the transformations
    combined_rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

    # Apply transformation based on object type
    if isinstance(obj, Sphere):
        obj_copy = Sphere(obj.center.copy(), obj.radius, obj.color)

        obj_copy.center = np.dot(np.append(obj_copy.center, 1), combined_rotation_matrix.T)[:3]
        obj_copy.center = np.dot(np.append(obj_copy.center, 1), translation_matrix.T)[:3]
        return obj_copy
    elif isinstance(obj, Plane):
        obj_copy = Plane(obj.center.copy(), obj.normal.copy(), obj.color)

        obj_copy.center = np.dot(np.append(obj_copy.center, 1), translation_matrix.T)[:3]
        obj_copy.center = np.dot(np.append(obj_copy.center, 1), combined_rotation_matrix.T)[:3]
        obj_copy.normal = np.dot(np.append(obj_copy.normal, 0), combined_rotation_matrix.T)[:3]
        return obj_copy
    elif isinstance(obj, Triangles):
        obj_copy = Triangles(obj.num_triangles, obj.num_vertices, obj.vertices.copy(), obj.triangle_index.copy(), obj.color)
        for i, triangle in enumerate(obj_copy.triangles):
            triangle.point1 = np.dot(np.append(triangle.point1, 1), translation_matrix.T)[:3]
            triangle.point1 = np.dot(np.append(triangle.point1, 1), combined_rotation_matrix.T)[:3]
            triangle.point2 = np.dot(np.append(triangle.point2, 1), translation_matrix.T)[:3]
            triangle.point2 = np.dot(np.append(triangle.point2, 1), combined_rotation_matrix.T)[:3]
            triangle.point3 = np.dot(np.append(triangle.point3, 1), translation_matrix.T)[:3]
            triangle.point3 = np.dot(np.append(triangle.point3, 1), combined_rotation_matrix.T)[:3]
            triangle.normal = np.cross(triangle.point1 - triangle.point2, triangle.point1 - triangle.point3)
            obj_copy.triangles[i] = triangle
        return obj_copy
    else:
        raise ValueError("Unsupported object type")

main()
