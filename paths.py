from typing import List, Tuple
import heapq
import numpy as np


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


# Функция для вычисления длины маршрута с учетом ландшафта
def calculate_path_length(path: List[Tuple[int, int]], _terrain_map: np.ndarray) -> float:
    length = 0
    for j in range(len(path) - 1):
        x1, y1 = path[j]
        x2, y2 = path[j + 1]
        terrain_cost = abs(_terrain_map[x2, y2] - _terrain_map[x1, y1])
        length += heuristic((x1, y1), (x2, y2)) + terrain_cost
    return length


# Поиск пути с использованием A*
def a_star_path(_start: Tuple[int, int], _end: Tuple[int, int], _terrain_map: np.ndarray) -> List[Tuple[int, int]]:
    open_list: List[Node] = []
    closed_list: set[Tuple[int, int]] = set()

    start_node: Node = Node(_start, 0, heuristic(_start, _end))
    heapq.heappush(open_list, start_node)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while open_list:
        current_node: Node = heapq.heappop(open_list)
        if current_node.position == _end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_list.add(current_node.position)

        for neighbor in directions:
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


def dynamic_programming_path(_start: Tuple[int, int], _end: Tuple[int, int], _terrain_map: np.ndarray) -> List[
    Tuple[int, int]]:
    # Размеры карты
    rows, cols = _terrain_map.shape

    # Таблица для хранения минимальной стоимости пути
    dp = np.full((rows, cols), float('inf'))
    dp[_start[0], _start[1]] = 0  # Стоимость старта 0

    # Таблица для восстановления пути
    parent = np.full((rows, cols), None)

    # Очередь для обхода карты (по принципу волнового алгоритма)
    queue = [(0, _start)]  # (стоимость, точка)

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Алгоритм динамического программирования
    while queue:
        current_cost, (x, y) = heapq.heappop(queue)

        # Если достигли конечной точки, восстанавливаем путь
        if (x, y) == _end:
            path = []
            while (x, y) != _start:
                path.append((x, y))
                x, y = parent[x, y]
            path.append(_start)
            return path[::-1]  # Путь в обратном порядке

        # Проверяем соседей
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:  # Проверяем, что координаты в пределах карты
                new_cost = current_cost + abs(_terrain_map[x, y] - _terrain_map[nx, ny])
                if new_cost < dp[nx, ny]:  # Если нашли более дешевый путь
                    dp[nx, ny] = new_cost
                    parent[nx, ny] = (x, y)
                    heapq.heappush(queue, (new_cost, (nx, ny)))

    return []  # Если путь не найден
