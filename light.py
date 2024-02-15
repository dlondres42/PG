import numpy as np

class Light:
    def __init__(self, position: np.array, intensity: tuple):
        self.position = position
        self.intensity = intensity

    def __str__(self):
        return f"Light(position={self.position}, intensity={self.intensity})"

    def __repr__(self):
        return str(self)