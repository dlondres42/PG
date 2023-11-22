from vector import Vector

class Scene:
    def __init__(self, camera: Vector, objects, width, height) -> None:
        self.camera = camera
        self.objects = objects
        self.width = width
        self.height = height
