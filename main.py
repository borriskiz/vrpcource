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
# Список пунктов маршрута

start: Tuple[int, int] = (100, 300)  # Начальная точка
points: List[Tuple[int, int]] = [(220, 414), (384, 273), (106, 271), (290, 232), (301, 410)]  # Пункты для маршрутизации


# Генерация карты
def generate_map(_width: int, _height: int, _scale: float, _octaves: int, _persistence: float,
                 _lacunarity: float) -> np.ndarray:
    map_data: np.ndarray = np.zeros((_height, _width))

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
terrain_map: np.ndarray = generate_map(width, height, scale, octaves, persistence, lacunarity)

# Визуализация исходного ландшафта
plt.imshow(terrain_map, cmap='terrain')
plt.colorbar()
plt.show()


# Класс для узлов в графе
class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float, h_cost: float, parent: 'Node' = None) -> None:
        self.position: Tuple[int, int] = position
        self.g_cost: float = g_cost  # Стоимость пути от стартовой точки
        self.h_cost: float = h_cost  # Эвристическая стоимость (расстояние до цели)
        self.f_cost: float = g_cost + h_cost  # Общая стоимость
        self.parent: 'Node' = parent

    def __lt__(self, other: 'Node') -> bool:
        return self.f_cost < other.f_cost


# Эвристическая функция (поиск по прямой линии)
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Функция поиска пути с использованием A*
def a_star(_start: Tuple[int, int], _end: Tuple[int, int], _terrain_map: np.ndarray) -> Tuple[Tuple[
    int, int], ...] | None:
    open_list: List[Node] = []
    closed_list: set[Tuple[int, int]] = set()

    # Начальный узел
    start_node: Node = Node(_start, 0, heuristic(_start, _end))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node: Node = heapq.heappop(open_list)
        if current_node.position == _end:
            # Восстановление пути
            _path: List[Tuple[int, int]] = []
            while current_node:
                _path.append(current_node.position)
                current_node = current_node.parent
            return tuple(_path[::-1])

        closed_list.add(current_node.position)

        # Проверка соседей (вверх, вниз, влево, вправо)
        for neighbor in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor_pos: Tuple[int, int] = (
                current_node.position[0] + neighbor[0], current_node.position[1] + neighbor[1])
            if 0 <= neighbor_pos[0] < _terrain_map.shape[0] and 0 <= neighbor_pos[1] < _terrain_map.shape[1]:
                if neighbor_pos in closed_list:
                    continue

                # Стоимость перехода с учетом перепада высоты
                g_cost: float = current_node.g_cost + abs(_terrain_map[neighbor_pos[0], neighbor_pos[1]] - _terrain_map[
                    current_node.position[0], current_node.position[1]])

                h_cost: float = heuristic(neighbor_pos, _end)
                neighbor_node: Node = Node(neighbor_pos, g_cost, h_cost, current_node)
                heapq.heappush(open_list, neighbor_node)

    return None  # Путь не найден


# Функция для маршрутизации с методом ближайшего соседа
def nearest_neighbor_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _terrain_map: np.ndarray) -> List[
    Tuple[int, int]]:
    unvisited_points = _points.copy()  # Копируем список точек для посещения
    current_point = _start  # Начальная точка
    total_path = [_start]  # Начальный маршрут

    # Пока есть не посещенные точки
    while unvisited_points:
        # Выбираем ближайшую точку
        closest_point = min(unvisited_points, key=lambda _point: a_star(current_point, _point, _terrain_map)[-1][0])
        # Находим путь от текущей точки до ближайшей
        _path = a_star(current_point, closest_point, _terrain_map)
        if _path:
            total_path.extend(_path[1:])  # Добавляем путь (без начальной точки, она уже в маршруте)
            current_point = closest_point
            unvisited_points.remove(closest_point)  # Удаляем из списка не посещенных точек

    return total_path


# Поиск маршрута между точками
path: List[Tuple[int, int]] = nearest_neighbor_routing(start, points, terrain_map)

# Визуализация
plt.imshow(terrain_map, cmap='terrain')
plt.colorbar()

# Рисуем все точки маршрута
for point in points:
    plt.scatter(point[1], point[0], color="purple", s=50, zorder=15)  # Все точки маршрута

# Рисуем путь поверх карты, не изменяя исходные значения карты
if path:
    # Рисуем линии и стрелки между узлами пути
    for i in range(len(path) - 1):
        start_point: Tuple[int, int] = path[i]
        end_point: Tuple[int, int] = path[i + 1]

        # Рисуем линию пути
        plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], color="red", linewidth=1, zorder=5)

        if i % 10 == 0:  # Стрелки каждые 10 точек
            plt.arrow(
                start_point[1], start_point[0],
                end_point[1] - start_point[1], end_point[0] - start_point[0],
                head_width=8, head_length=12, fc="blue", ec="blue", length_includes_head=True, zorder=10
            )

    # Выделяем начальную точку как большую зеленую точку
    plt.scatter(start[1], start[0], color="green", s=80, zorder=20)  # Начальная точка

# Показать финальный результат
plt.show()
