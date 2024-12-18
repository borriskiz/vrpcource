from paths import get_path_from_cache_or_calculate, get_path_length_from_cache_or_calculate
from typing import List, Tuple
import numpy as np

path_cache = {}
path_length_cache = {}

# Счетчики для подсчета проверок
path_check_count = 0  # Для подсчета проверок путей


# Функция маршрутизации методом ближайшего соседа
def nearest_neighbor_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                             _terrain_map: np.ndarray) -> List[Tuple[int, int]]:
    global path_check_count

    unvisited_points = _points.copy()
    current_point = _start
    final_path = [_start]

    while unvisited_points:
        # Находим ближайшую точку с учетом кэширования длины пути
        closest_point = min(unvisited_points, key=lambda p: get_path_length_from_cache_or_calculate(
            current_point, p, _terrain_map, path_cache, path_length_cache))
        path_check_count += len(unvisited_points)

        # Получаем путь между текущей точкой и ближайшей с учётом кэширования
        path_segment = get_path_from_cache_or_calculate(current_point, closest_point, _terrain_map, path_cache,
                                                        path_length_cache)

        # Увеличиваем счетчик проверок путей
        path_check_count += 1

        final_path.extend(path_segment[1:])  # Добавляем путь без начальной точки сегмента

        # Обновляем текущую точку и убираем её из не посещённых
        current_point = closest_point
        unvisited_points.remove(closest_point)

    # Добавляем путь к конечной точке с учетом кэширования
    final_path_segment = get_path_from_cache_or_calculate(current_point, _end, _terrain_map, path_cache,
                                                          path_length_cache)

    # Увеличиваем счетчик проверок пути и длины пути
    path_check_count += 1

    final_path.extend(final_path_segment[1:])

    # Выводим количество проверок
    print(f"Количество проверок путей: {path_check_count}")
    print(f"Количество реальных вычислений путей: {len(path_length_cache)}")

    path_cache.clear()  # Очищаем кэш путей после завершения маршрута
    path_length_cache.clear()  # Очищаем кэш длин путей после завершения маршрута

    return final_path
