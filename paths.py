from typing import List, Tuple, Dict
from map import plot_visited_nodes
import heapq
import numpy as np

height_weight: float = 200


# Функция для вычисления длины маршрута с учетом ландшафта
def calculate_path_length(path: List[Tuple[int, int]], _terrain_map: np.ndarray) -> float:
    length = 0
    for j in range(len(path) - 1):
        x1, y1 = path[j]
        x2, y2 = path[j + 1]
        terrain_cost = calculate_height((x1, y1), (x2, y2), _terrain_map)
        length += heuristic((x1, y1), (x2, y2)) + terrain_cost
    return length


# Получение пути из кэша (с добавлением кэша длины пути)
def get_path_from_cache_or_calculate(start: Tuple[int, int], end: Tuple[int, int], terrain_map: np.ndarray,
                                     _path_cache: dict, _path_length_cache: dict) -> List[Tuple[int, int]]:
    # Проверка, есть ли уже путь в кэше
    if (start, end) in _path_cache:
        return _path_cache[(start, end)]

    # Если пути нет в кэше, вычисляем его с использованием A* и сохраняем в кэш
    path = a_star_path(start, end, terrain_map)
    _path_cache[(start, end)] = path

    # Также вычисляем длину пути и сохраняем ее в кэш
    path_length = calculate_path_length(path, terrain_map)
    _path_length_cache[(start, end)] = path_length

    return path


# Получение длины пути из кэша (если есть)
def get_path_length_from_cache_or_calculate(start: Tuple[int, int], end: Tuple[int, int], terrain_map: np.ndarray,
                                            _path_cache: dict, _path_length_cache: dict) -> float:
    # Проверка, есть ли длина пути в кэше
    if (start, end) in _path_length_cache:
        return _path_length_cache[(start, end)]

    # Если длина пути нет в кэше, получаем путь из кэша или вычисляем его
    path = get_path_from_cache_or_calculate(start, end, terrain_map, _path_cache, _path_length_cache)

    # Рассчитываем и сохраняем длину пути в кэш
    path_length = calculate_path_length(path, terrain_map)
    _path_length_cache[(start, end)] = path_length

    return path_length


# Класс для узлов в графе
class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float = 0.0, h_cost: float = 0.0,
                 parent: 'Node' = None) -> None:
        self.position: Tuple[int, int] = position
        self.g_cost: float = g_cost  # Стоимость пути от старта до текущего узла
        self.h_cost: float = h_cost  # Эвристическая стоимость от текущего узла до цели
        self.f_cost: float = g_cost + h_cost  # Общая стоимость (g + h)
        self.parent: 'Node' = parent

    def __lt__(self, other: 'Node') -> bool:
        return self.f_cost < other.f_cost

    def __eq__(self, other: 'Node') -> bool:
        return self.position == other.position


# Эвристическая функция (расстояние Евклида)
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    # return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + calculate_height(a, b, _terrain_map) ** 2)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Функция для вычисления высоты между точками
def calculate_height(_from: Tuple[int, int], _to: Tuple[int, int], _terrain_map: np.ndarray) -> float:
    height1 = float(_terrain_map[_from[0], _from[1]])
    height2 = float(_terrain_map[_to[0], _to[1]])
    return height_weight * abs(height2 - height1)


# Основная функция поиска пути с использованием A*
def a_star_path(start: Tuple[int, int], end: Tuple[int, int], _terrain_map: np.ndarray,
                _show_visited_nodes: bool = False) -> List[Tuple[int, int]]:
    open_list: List[Node] = []  # Очередь с приоритетами
    closed_set: set[Tuple[int, int]] = set()  # Множество посещенных узлов
    came_from: Dict[Tuple[int, int], Node] = {}  # Для восстановления пути
    g_costs: Dict[Tuple[int, int], float] = {}  # Стоимости пути для каждого узла

    start_node: Node = Node(start, 0, heuristic(start, end))
    heapq.heappush(open_list, start_node)
    g_costs[start] = 0  # Начальная стоимость пути для старта

    # Направления для поиска соседей (включая диагональные)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    path = []  # Список для хранения пути

    while open_list:
        current_node: Node = heapq.heappop(open_list)

        # Если нашли путь до цели
        if current_node.position == end:
            # Восстанавливаем путь от конечного узла до начального
            while current_node:
                path.append(current_node.position)
                current_node = came_from.get(current_node.position)
            if _show_visited_nodes:
                plot_visited_nodes(start, end, closed_set, _terrain_map)
            return path[::-1]

        closed_set.add(current_node.position)

        for neighbor in directions:
            neighbor_pos = (current_node.position[0] + neighbor[0], current_node.position[1] + neighbor[1])

            # Проверка, что сосед в пределах карты
            if 0 <= neighbor_pos[0] < _terrain_map.shape[0] and 0 <= neighbor_pos[1] < _terrain_map.shape[1]:
                if neighbor_pos in closed_set:
                    continue

                # Вычисление новых g и h стоимостей для соседа
                g_cost = current_node.g_cost + calculate_height(current_node.position, neighbor_pos, _terrain_map)
                h_cost = heuristic(neighbor_pos, end)

                # Если сосед уже в открытом списке с меньшей или равной стоимостью, пропускаем его
                if neighbor_pos in g_costs and g_costs[neighbor_pos] <= g_cost:
                    continue

                g_costs[neighbor_pos] = g_cost
                neighbor_node = Node(neighbor_pos, g_cost, h_cost, current_node)
                heapq.heappush(open_list, neighbor_node)
                came_from[neighbor_pos] = current_node  # Отслеживаем путь

    return []
