from image import Image
from scene import Scene
from ray import Ray
from image import Image
from vector import Point, Color
from scene import Scene

INF = int(2e9)


class Engine:
    def render(self, scene: Scene) -> Image:
        hres = scene.width
        vres = scene.height
        camera = scene.camera
        C, w, u, v, d = camera.params()
        _x1, _x2, _x3 = w.dot_product(u), w.dot_product(v), u.dot_product(v)
        # w, u, v -> Base ortonormal

        O = C + (w * d)
        tam_x, tam_y = 1, 1
        desl_h = ((2 * tam_x) / (hres - 1)) * u
        desl_v = ((2 * tam_y) / (vres - 1)) * v
        vet_inicial = w * d - (tam_x * u) - (tam_y * v)

        camera = scene.camera
        pixels = Image(hres, vres)

        for j in range(vres):
            for i in range(hres):
                v_r = vet_inicial + (i * desl_h) + (j * desl_v)
                ray = Ray(C, v_r)
                _, color = self.ray_trace(ray, scene)
                pixels.set_pixel(i, j, color)

        return pixels

    def find_nearest(self, ray: Ray, scene: Scene):
        t_min = INF
        obj_hit = None
        hit_point = None

        for obj in scene.objects:
            t = obj.intersection(ray)
            if t < t_min:
                hit_point = ray.origin + (ray.direction * t)
                obj_hit = obj
                t_min = t
        return (hit_point, obj_hit)

    def ray_trace(self, ray: Ray, scene: Scene) -> Color:
        color = Color(0, 255, 0)

        hit_point, obj_hit = self.find_nearest(ray, scene)
        if obj_hit is not None:
            color = obj_hit.color
        return hit_point, color
