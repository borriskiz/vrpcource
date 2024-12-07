from typing import List, Tuple
import heapq
import numpy as np


# Класс для узлов в графе
class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float = 0.0, h_cost: float = 0.0,
                 parent: 'Node' = None) -> None:
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


# Функция для вычисления длины маршрута с учетом ландшафта
def calculate_path_length(path: List[Tuple[int, int]], _terrain_map: np.ndarray) -> float:
    length = 0
    for j in range(len(path) - 1):
        x1, y1 = path[j]
        x2, y2 = path[j + 1]
        terrain_cost = calculate_height((x1, y1), (x2, y2), _terrain_map)
        length += heuristic((x1, y1), (x2, y2)) + terrain_cost
    return length


# Функция для вычисления длины маршрута с учетом ландшафта
def calculate_height(_from: Tuple[int, int], _to: Tuple[int, int], _terrain_map: np.ndarray) -> float:
    height1: float = float(_terrain_map[_from[0], _from[1]])
    height2: float = float(_terrain_map[_to[0], _to[1]])
    return 150 * abs(height2 - height1)


# Поиск пути с использованием A*
def a_star_path(_start: Tuple[int, int], _end: Tuple[int, int], _terrain_map: np.ndarray) -> List[Tuple[int, int]]:
    open_list: List[Node] = []
    closed_list: set[Tuple[int, int]] = set()
    came_from: dict[Tuple[int, int], Node] = {}  # Словарь для отслеживания пути
    g_costs: dict[Tuple[int, int], float] = {}  # Словарь для хранения g_cost узлов

    start_node: Node = Node(_start, 0, heuristic(_start, _end))
    heapq.heappush(open_list, start_node)
    g_costs[_start] = 0  # Начальная стоимость пути для старта
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while open_list:
        current_node: Node = heapq.heappop(open_list)

        if current_node.position == _end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = came_from.get(current_node.position)
            return path[::-1]

        closed_list.add(current_node.position)

        for neighbor in directions:
            neighbor_pos = (current_node.position[0] + neighbor[0], current_node.position[1] + neighbor[1])
            if 0 <= neighbor_pos[0] < _terrain_map.shape[0] and 0 <= neighbor_pos[1] < _terrain_map.shape[1]:
                if neighbor_pos in closed_list:
                    continue

                # Вычисляем новые g_cost и h_cost
                g_cost = current_node.g_cost + calculate_height(current_node.position, neighbor_pos, _terrain_map)
                h_cost = heuristic(neighbor_pos, _end)

                # Проверяем, если этот сосед уже в открытом списке с более высокой стоимостью, пропускаем его
                if neighbor_pos in g_costs and g_costs[neighbor_pos] <= g_cost:
                    continue

                g_costs[neighbor_pos] = g_cost  # Обновляем g_cost для соседа
                neighbor_node = Node(neighbor_pos, g_cost, h_cost, current_node)
                heapq.heappush(open_list, neighbor_node)
                came_from[neighbor_pos] = current_node  # Отслеживаем путь

    return []
