from paths import heuristic, get_path_from_cache_or_calculate
from typing import List, Tuple
import numpy as np

path_cache = {}


# Функция маршрутизации методом ближайшего соседа
def nearest_neighbor_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                             _terrain_map: np.ndarray) -> List[Tuple[int, int]]:
    unvisited_points = _points.copy()
    current_point = _start
    total_path = [_start]

    while unvisited_points:
        # Находим ближайшую точку
        closest_point = min(unvisited_points, key=lambda p: heuristic(current_point, p))

        # Получаем путь между текущей точкой и ближайшей с учётом кэширования
        path_segment = get_path_from_cache_or_calculate(current_point, closest_point, _terrain_map, path_cache)
        total_path.extend(path_segment[1:])  # Добавляем путь без начальной точки сегмента

        # Обновляем текущую точку и убираем её из не посещённых
        current_point = closest_point
        unvisited_points.remove(closest_point)

    # Добавляем путь к конечной точке
    final_path_segment = get_path_from_cache_or_calculate(current_point, _end, _terrain_map, path_cache)
    total_path.extend(final_path_segment[1:])

    return total_path
