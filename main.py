import numpy as np
import noise
import heapq
import matplotlib.pyplot as plt
from typing import List, Tuple

# Параметры карты
width: int = 512
height: int = 512
scale: float = 100.0  # Масштаб шума

# Параметры шума
octaves: int = 8  # Количество октав
persistence: float = 0.5  # Влияние каждой октавы
lacunarity: float = 2.0  # Частота осцилляций

# Параметры маршрута
start: Tuple[int, int] = (100, 300)  # Начальная точка
end: Tuple[int, int] = (400, 100)  # Конечная точка
points: List[Tuple[int, int]] = [(220, 414), (384, 273), (106, 271), (290, 232), (301, 410)]  # Пункты маршрута


# Генерация карты
def generate_map(_width: int, _height: int, _scale: float, _octaves: int, _persistence: float,
                 _lacunarity: float) -> np.ndarray:
    map_data: np.ndarray = np.zeros((_height, _width))
    for y in range(_height):
        for x in range(_width):
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


# Класс для узлов в графе
class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float, h_cost: float, parent: 'Node' = None) -> None:
        self.position: Tuple[int, int] = position
        self.g_cost: float = g_cost
        self.h_cost: float = h_cost
        self.f_cost: float = g_cost + h_cost
        self.parent: 'Node' = parent

    def __lt__(self, other: 'Node') -> bool:
        return self.f_cost < other.f_cost


# Эвристическая функция
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Поиск пути с использованием A*
def a_star(_start: Tuple[int, int], _end: Tuple[int, int], _terrain_map: np.ndarray) -> List[Tuple[int, int]]:
    open_list: List[Node] = []
    closed_list: set[Tuple[int, int]] = set()

    start_node: Node = Node(_start, 0, heuristic(_start, _end))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node: Node = heapq.heappop(open_list)
        if current_node.position == _end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_list.add(current_node.position)

        for neighbor in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor_pos = (current_node.position[0] + neighbor[0], current_node.position[1] + neighbor[1])
            if 0 <= neighbor_pos[0] < _terrain_map.shape[0] and 0 <= neighbor_pos[1] < _terrain_map.shape[1]:
                if neighbor_pos in closed_list:
                    continue

                g_cost = current_node.g_cost + abs(
                    float(_terrain_map[neighbor_pos[0], neighbor_pos[1]]) -
                    float(_terrain_map[current_node.position[0], current_node.position[1]])
                )
                h_cost = heuristic(neighbor_pos, _end)
                neighbor_node = Node(neighbor_pos, g_cost, h_cost, current_node)
                heapq.heappush(open_list, neighbor_node)
    return []


# Функция маршрутизации методом ближайшего соседа
def nearest_neighbor_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                             _terrain_map: np.ndarray) -> List[Tuple[int, int]]:
    unvisited_points = _points.copy()
    current_point = _start
    total_path = [_start]

    while unvisited_points:
        closest_point = min(unvisited_points, key=lambda p: heuristic(current_point, p))
        path_segment = a_star(current_point, closest_point, _terrain_map)
        total_path.extend(path_segment[1:])
        current_point = closest_point
        unvisited_points.remove(closest_point)

    # Добавляем путь к конечной точке
    final_path_segment = a_star(current_point, _end, _terrain_map)
    total_path.extend(final_path_segment[1:])
    return total_path


# Генерация карты
terrain_map: np.ndarray = generate_map(width, height, scale, octaves, persistence, lacunarity)

# Поиск маршрута
path_nearest_neighbor: List[Tuple[int, int]] = nearest_neighbor_routing(start, points, end, terrain_map)

# Визуализация исходного ландшафта
plt.figure(figsize=(10, 10))
plt.imshow(terrain_map, cmap='terrain')
plt.colorbar(label="Высота")
plt.scatter([p[1] for p in points], [p[0] for p in points], color="purple", s=50, label="Точки маршрута")
plt.scatter(start[1], start[0], color="green", s=80, label="Начальная точка")
plt.scatter(end[1], end[0], color="red", s=80, label="Конечная точка")
plt.title("Оригинальный ландшафт с узлами")
plt.legend()
plt.show()

# Визуализация пути
plt.figure(figsize=(10, 10))
plt.imshow(terrain_map, cmap='terrain')
plt.colorbar(label="Высота")
plt.scatter([p[1] for p in points], [p[0] for p in points], color="purple", s=50, label="Точки маршрута", zorder=10)
plt.scatter(start[1], start[0], color="green", s=80, label="Начальная точка", zorder=10)
plt.scatter(end[1], end[0], color="red", s=80, label="Конечная точка", zorder=10)
for i in range(len(path_nearest_neighbor) - 1):
    plt.plot([path_nearest_neighbor[i][1], path_nearest_neighbor[i + 1][1]],
             [path_nearest_neighbor[i][0], path_nearest_neighbor[i + 1][0]], color="red", linewidth=1, zorder=2)
    if i % 30 == 0:  # Рисуем стрелки каждые 10 шагов
        plt.arrow(
            path_nearest_neighbor[i][1], path_nearest_neighbor[i][0],
            path_nearest_neighbor[i + 1][1] - path_nearest_neighbor[i][1],
            path_nearest_neighbor[i + 1][0] - path_nearest_neighbor[i][0],
            head_width=5, head_length=8, fc="blue", ec="blue", zorder=3
        )
plt.title("Алгоритм ближайшего соседа")
plt.legend()
plt.show()
