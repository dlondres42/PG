from camera import Camera


class Scene:
    def __init__(self, camera: Camera, objects, width, height) -> None:
        self.camera = camera
        self.objects = objects
        self.width = width
        self.height = height
