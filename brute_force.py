import itertools
from typing import List, Tuple
import numpy as np
from paths import calculate_path_length, get_path_from_cache_or_calculate

path_cache = {}


# Функция для маршрутизации методом полного перебора всех маршрутов с промежуточными точками
def brute_force_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                        _terrain_map: np.ndarray) -> List[Tuple[int, int]]:
    # Объединяем начальную и конечную точку с точками маршрута
    all_points = [_start] + _points + [_end]

    # Проверяем количество точек и предупреждаем, если их слишком много
    if len(all_points) > 10:
        print("Внимание! Число точек больше 10, алгоритм будет работать очень долго!")

    best_path = None
    min_cost = float('inf')

    # Перебор всех перестановок
    for perm in itertools.permutations(_points):
        current_path = [_start] + list(perm) + [_end]
        total_cost = 0
        full_path = []

        # Проходим по всем сегментам маршрута
        for i in range(len(current_path) - 1):
            # Для каждого сегмента пути между точками вычисляем путь с учетом ландшафта
            segment_path = get_path_from_cache_or_calculate(current_path[i], current_path[i + 1], _terrain_map,
                                                            path_cache)
            # Добавляем сегмент пути в общий путь
            if i != len(current_path) - 2:  # не добавляем конечную точку повторно
                full_path.extend(segment_path[:-1])
            else:
                full_path.extend(segment_path)  # Для последнего сегмента включаем конечную точку

            # Рассчитываем стоимость сегмента
            total_cost += calculate_path_length(segment_path, _terrain_map)

        # Если найден новый минимальный путь, сохраняем его
        if total_cost < min_cost:
            min_cost = total_cost
            best_path = full_path

    return best_path