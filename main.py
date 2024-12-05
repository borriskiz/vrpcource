import numpy as np
import noise
import matplotlib.pyplot as plt
from numpy import ndarray

# Параметры карты
width = 512
height = 512
scale = 100.0  # Масштаб шума

# Параметры шума
octaves = 8  # Количество октав
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

import heapq
import matplotlib.pyplot as plt


# Определение класса для узлов в графе
class Node:
    def __init__(self, position, g_cost, h_cost, parent=None):
        self.position = position
        self.g_cost = g_cost  # Стоимость пути от стартовой точки
        self.h_cost = h_cost  # Эвристическая стоимость (расстояние до цели)
        self.f_cost = g_cost + h_cost  # Общая стоимость
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost


# Функция эвристики (поиск по прямой линии)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Функция поиска пути с использованием A*
def a_star(_start, _end, _terrain_map):
    open_list = []
    closed_list = set()

    # Начальный узел
    start_node = Node(_start, 0, heuristic(_start, _end))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.position == _end:
            # Восстановление пути
            _path = []
            while current_node:
                _path.append(current_node.position)
                current_node = current_node.parent
            return _path[::-1]

        closed_list.add(current_node.position)

        # Проверка соседей (вверх, вниз, влево, вправо)
        for neighbor in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor_pos = (current_node.position[0] + neighbor[0], current_node.position[1] + neighbor[1])
            if 0 <= neighbor_pos[0] < _terrain_map.shape[0] and 0 <= neighbor_pos[1] < _terrain_map.shape[1]:
                if neighbor_pos in closed_list:
                    continue

                # Стоимость перехода с учетом перепада высоты
                g_cost = current_node.g_cost + abs(_terrain_map[neighbor_pos[0], neighbor_pos[1]] - _terrain_map[
                    current_node.position[0], current_node.position[1]])
                h_cost = heuristic(neighbor_pos, _end)
                neighbor_node = Node(neighbor_pos, g_cost, h_cost, current_node)
                heapq.heappush(open_list, neighbor_node)

    return None  # Путь не найден


# Пример начальной и конечной точки
start = (0, 0)  # Начальная точка
end = (511, 511)  # Конечная точка

# Поиск пути между точками
path = a_star(start, end, terrain_map)

# Визуализация
terrain_map_copy = terrain_map.copy()  # Создаем копию карты, чтобы не менять оригинал

# Отметим путь на копии карты
if path:
    for i in range(len(path)):
        point = path[i]
        # Отметим путь на карте более выразительно, например, сделаем его ярким красным
        terrain_map_copy[point] = 2  # Используем значение, которое не присутствует в исходной карте для выделения пути

# Визуализация карты с путем
plt.imshow(terrain_map_copy, cmap='terrain')
plt.colorbar()

# Дополнительно, можно нарисовать путь более выразительным, увеличив его толщину
for i in range(len(path) - 1):
    start_point = path[i]
    end_point = path[i + 1]
    plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], color="red", linewidth=2)

plt.show()
