import numpy as np
import noise
import matplotlib.pyplot as plt

# Параметры карты
width = 512
height = 512
scale = 100.0  # Масштаб шума

# Параметры шума
octaves = 6  # Количество октав
persistence = 0.5  # Влияние каждой октавы
lacunarity = 2.0  # Частота осцилляций


# Генерация карты
def generate_map(width, height, scale, octaves, persistence, lacunarity):
    map_data = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            # Применение перлин-шума
            map_data[y][x] = noise.pnoise2(
                x / scale,
                y / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
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
