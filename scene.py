from camera import Camera


class Scene:
    def __init__(self, camera: Camera, objects, width, height, ambient_light=(0, 0, 0),  lights=None) -> None:
        self.camera = camera
        self.objects = objects
        self.ambient_light = ambient_light
        self.lights = lights if lights is not None else []
        self.width = width
        self.height = height
