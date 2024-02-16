import numpy as np
from camera import Camera
from color import Color
from objects import Sphere, Plane, Triangles, Material
from light import Light
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

    #test = Triangles(2, 4, np.array([[2, -0.3, 0], [2.5,0.1,0.2], [1.7, 0.5, -0.1], [1.5,0.3,1.2]]), [(0,1,2), (0,2,3)], Color(0, 255, 0))
    camera = Camera(O, w, u, v, dist)

    objects = [
        Sphere(np.array([2, 0, 0]), 0.2, Color(0, 255, 0), Material(kd=(0.5,0.5,0.5), ks=(0.25,0.25,0.25), ka=(0.2,0.2,0.2))),
        #affine_transform(Plane(np.array([1,-0.5,0]), np.array([0,1,0]), Color(0,0,255), Material(ka=(1,1,1))), translation=(0,0,0), rotation_angles=(0,0,0)),
        affine_transform(Sphere(np.array([2,0,0]), 0.3, Color(0,0,255), Material(kd=(0.5,0.5,0.5), ks=(0.25,0.25,0.25), ka=(0.2,0.2,0.2))), translation=(0,0,0), rotation_angles=(0,0,0)),
        # Triangles(2, 4, np.array([[4, 1, 0], [4,1,1], [4, 1, -1], [4,0,0]]), [(0,1,3), (1,2,3)], Color(250, 70, 55), Material(ka=(1,1,1)))
    ]

    ambient_light = (10, 10, 10)

    lights = [
        Light(np.array([0, 5, 5]), np.array([255, 223, 142])),
        Light(np.array([5, 5, 5]), np.array([255, 255, 255]))
    ]

    scene = Scene(camera, objects, hres, vres, ambient_light, lights)

    mtx = render(scene)
    image = Image.fromarray(mtx)
    image.save("output.png")  # save img
    # image.show()  # show img


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


def reflect_ray(L, normal):
    """ 
    Function used in ray_color in order to compute the
    reflector vector R for the specular component

    """
    # Compute the dot product of normal and L
    dot_product = np.dot(normal, L)

    # Compute the reflection vector R
    R = 2 * dot_product * normal - L

    return R

def ray_color(ray_origin, ray_direction, scene: Scene):
    t_min, obj_hit = find_nearest(ray_origin, ray_direction, scene)
    color = Color(0, 0, 0)
    if obj_hit is not None:
        ka = np.array(obj_hit.material.ka) # Ambient coefficient
        kd = np.array(obj_hit.material.kd) # Diffuse coefficient
        ks = np.array(obj_hit.material.ks) # Specular coefficient
        n = np.array(obj_hit.material.eta) # Roughness coefficient

        Ia = np.array(scene.ambient_light)

        intersection_point = ray_origin + t_min * ray_direction

        
        if isinstance(obj_hit, Sphere):
            normal = normalize(intersection_point - obj_hit.center)
        elif isinstance(obj_hit, Plane):
            normal = obj_hit.normal
        elif isinstance(obj_hit, Triangles):
            normal = obj_hit.normal_at(intersection_point)  

        # View vector (direction towards the camera)
        V = normalize(scene.camera.origin - intersection_point)

        # Initialize the total color contribution
        total_color = np.array([0,0,0])
        total_color = total_color.astype(np.float64)

        # Ambient component
        ambient_color = ka * Ia
        total_color += ambient_color

        for light in scene.lights:
            # Direction from the intersection point to the light
            L = normalize(light.position - intersection_point)

            # Diffuse component
            diffuse_dot = max(np.dot(normal, L), 0)
            diffuse_color = kd * light.intensity * diffuse_dot
            total_color += diffuse_color

            # Specular component
            R = reflect_ray(L, normal)
            specular_dot = max(np.dot(R, V), 0)
            specular_color = ks * light.intensity * (specular_dot ** n)
            total_color += specular_color

            color_tuple = (obj_hit.color.r, obj_hit.color.g, obj_hit.color.b)
            color_array = np.array(color_tuple) / 255

            # Perform element-wise multiplication
            total_color *= color_array

        r, g, b = total_color.clip(0, 255)
        color = Color(r, g, b)
        #color = obj_hit.color
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
