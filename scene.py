from camera import Camera


class Scene:
    def __init__(self, camera: Camera, objects, width, height, lights=None) -> None:
        self.camera = camera
        self.objects = objects
        self.lights = lights if lights is not None else []
        self.width = width
        self.height = height
