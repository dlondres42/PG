import numpy as np

class Light:
    def __init__(self, position: np.array, intensity: np.array, radius=0):
        self.position = position
        self.intensity = intensity
        self.radius = radius  # Represents the size of the light source for soft shadows

    def __str__(self):
        return f"Light(position={self.position}, intensity={self.intensity})"

    def __repr__(self):
        return str(self)