from paths import a_star_path, heuristic
from typing import List, Tuple
import numpy as np


# Функция маршрутизации методом ближайшего соседа
def nearest_neighbor_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                             _terrain_map: np.ndarray) -> List[Tuple[int, int]]:
    unvisited_points = _points.copy()
    current_point = _start
    total_path = [_start]

    while unvisited_points:
        closest_point = min(unvisited_points, key=lambda p: heuristic(current_point, p))
        path_segment = a_star_path(current_point, closest_point, _terrain_map)
        total_path.extend(path_segment[1:])
        current_point = closest_point
        unvisited_points.remove(closest_point)

    # Добавляем путь к конечной точке
    final_path_segment = a_star_path(current_point, _end, _terrain_map)
    total_path.extend(final_path_segment[1:])
    return total_path
