from image import Image
from scene import Scene
from ray import Ray
from image import Image
from vector import Vector, Point
from color import Color
from scene import Scene
from sphere import Sphere
class Engine:
    def render(self, scene: Scene) -> Image:
        width = scene.width
        height = scene.height
        aspect_ratio = width / height
        x0 = -1.
        x1 = 1.
        xstep = (x1 - x0) / (width - 1)
        y0 = -1. / aspect_ratio
        y1 = 1. / aspect_ratio
        ystep = (y1 - y0) / (height - 1)

        camera = scene.camera
        pixels = Image(width, height)

        for j in range(height):
            y = y0 + j * ystep
            for i in range(width):
                x = x0 + i * xstep
                ray = Ray(camera, Point(x, y) - camera)
                pixels.set_pixel(i, j, self.ray_trace(ray, scene))
        
        return pixels
    
    def find_nearest(self, ray: Ray, scene: Scene):
        dist_min = None
        obj_hit = None

        for obj in scene.objects:
            dist = obj.intersection(ray)
            if dist is not None and (obj_hit is None or dist < dist_min):
                dist_min = dist
                obj_hit = obj
            return (dist_min, obj_hit)

    def color_at(self, obj_hit, hit_pos: Point, scene: Scene) -> Color:
        return obj_hit.color

    def ray_trace(self, ray: Ray, scene: Scene) -> Color:
        color = Color(0,0,0)
        
        dist_hit, obj_hit = self.find_nearest(ray, scene)
        if obj_hit is None:
            return color
        hit_pos = ray.origin + (ray.direction * dist_hit)

        color += self.color_at(obj_hit, hit_pos, scene)
        return color
    