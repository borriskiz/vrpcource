import numpy as np
import noise
import matplotlib.pyplot as plt
from numpy import ndarray

# Параметры карты
width = 512
height = 512
scale = 100.0  # Масштаб шума

# Параметры шума
octaves = 6  # Количество октав
persistence = 0.5  # Влияние каждой октавы
lacunarity = 2.0  # Частота осцилляций


# Генерация карты
def generate_map(_width: int, _height: int, _scale: float, _octaves: float, _persistence: float,
                 _lacunarity: float) -> ndarray:
    map_data = np.zeros((_height, _width))

    for y in range(_height):
        for x in range(_width):
            # Применение перлин-шума
            map_data[y][x] = noise.pnoise2(
                x / _scale,
                y / _scale,
                octaves=_octaves,
                persistence=_persistence,
                lacunarity=_lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=42
            )

    return map_data


# Генерация карты
terrain_map = generate_map(width, height, scale, octaves, persistence, lacunarity)

# Визуализация
plt.imshow(terrain_map, cmap='terrain')
plt.colorbar()
plt.show()
