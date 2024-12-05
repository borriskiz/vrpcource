import numpy as np
import noise
import heapq
import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Tuple  # Импортируем Tuple из typing

# Параметры карты
width: int = 512
height: int = 512
scale: float = 100.0  # Масштаб шума

# Параметры шума
octaves: int = 8  # Количество октав
persistence: float = 0.5  # Влияние каждой октавы
lacunarity: float = 2.0  # Частота осцилляций


# Генерация карты
def generate_map(_width: int, _height: int, _scale: float, _octaves: int, _persistence: float,
                 _lacunarity: float) -> ndarray:
    map_data: ndarray = np.zeros((_height, _width))

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
terrain_map: ndarray = generate_map(width, height, scale, octaves, persistence, lacunarity)

# Визуализация исходного ландшафта
plt.imshow(terrain_map, cmap='terrain')
plt.colorbar()
plt.show()


# Определение класса для узлов в графе
class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float, h_cost: float, parent: 'Node' = None) -> None:
        self.position: Tuple[int, int] = position
        self.g_cost: float = g_cost  # Стоимость пути от стартовой точки
        self.h_cost: float = h_cost  # Эвристическая стоимость (расстояние до цели)
        self.f_cost: float = g_cost + h_cost  # Общая стоимость
        self.parent: 'Node' = parent

    def __lt__(self, other: 'Node') -> bool:
        return self.f_cost < other.f_cost


# Функция эвристики (поиск по прямой линии)
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Функция поиска пути с использованием A*
def a_star(_start: Tuple[int, int], _end: Tuple[int, int], _terrain_map: ndarray) -> Tuple[Tuple[int, int], ...] | None:
    open_list: list[Node] = []
    closed_list: set[Tuple[int, int]] = set()

    # Начальный узел
    start_node: Node = Node(_start, 0, heuristic(_start, _end))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node: Node = heapq.heappop(open_list)
        if current_node.position == _end:
            # Восстановление пути
            _path: list[Tuple[int, int]] = []
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
                    current_node.position[0], current_node.position[1]].item())

                h_cost: float = heuristic(neighbor_pos, _end)
                neighbor_node: Node = Node(neighbor_pos, g_cost, h_cost, current_node)
                heapq.heappush(open_list, neighbor_node)

    return None  # Путь не найден


# Пример начальной и конечной точки
start: Tuple[int, int] = (100, 300)  # Начальная точка
end: Tuple[int, int] = (300, 100)  # Конечная точка

# Поиск пути между точками
path: Tuple[Tuple[int, int], ...] | None = a_star(start, end, terrain_map)  # Визуализация

# Визуализация
plt.imshow(terrain_map, cmap='terrain')
plt.colorbar()

# Рисуем путь поверх карты, не изменяя исходные значения карты
if path:

    # Рисуем линии и стрелки между узлами пути
    for i in range(len(path) - 1):
        start_point: Tuple[int, int] = path[i]
        end_point: Tuple[int, int] = path[i + 1]

        # Рисуем линию пути
        plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], color="red", linewidth=1)

        if i % 30 == 0:
            # Добавляем стрелку, уменьшаем размер и немного увеличиваем расстояние
            plt.arrow(
                start_point[1], start_point[0],  # Координаты начала стрелки
                end_point[1] - start_point[1], end_point[0] - start_point[0],  # Смещение
                head_width=4, head_length=8, fc="black", ec="black", length_includes_head=True, zorder=10
                # Размер стрелок
            )
    # Выделяем начальную и конечную точки как большие точки
    plt.scatter(start[1], start[0], color="green", s=20, zorder=20)  # Начальная точка
    plt.scatter(end[1], end[0], color="black", s=20, zorder=20)  # Конечная точка

# Показать финальный результат
plt.show()
