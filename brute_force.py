import itertools
from typing import List, Tuple
import numpy as np
from paths import get_path_from_cache_or_calculate, get_path_length_from_cache_or_calculate

path_cache = {}
path_length_cache = {}


# Функция для маршрутизации методом полного перебора всех маршрутов с промежуточными точками
def brute_force_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                        _terrain_map: np.ndarray) -> List[Tuple[int, int]]:
    # Проверяем количество точек и предупреждаем, если их слишком много
    if len(_points) > 10:
        print("Внимание! Число точек больше 10, алгоритм будет работать очень долго!")

    final_path = None
    min_cost = float('inf')

    total_permutations = 0  # Считаем количество перестановок
    total_path_checks = 0  # Считаем количество вычислений путей

    # Перебор всех перестановок
    for perm in itertools.permutations(_points):
        total_permutations += 1  # Увеличиваем счетчик перестановок
        current_path = [_start] + list(perm) + [_end]
        total_cost = 0
        full_path = []

        # Проходим по всем сегментам маршрута
        for i in range(len(current_path) - 1):
            # Для каждого сегмента пути между точками вычисляем путь с учетом ландшафта
            segment_path = get_path_from_cache_or_calculate(current_path[i], current_path[i + 1], _terrain_map,
                                                            path_cache, path_length_cache)
            total_path_checks += 1  # Увеличиваем счетчик вычислений путей

            # Добавляем сегмент пути в общий путь
            if i != len(current_path) - 2:  # не добавляем конечную точку повторно
                full_path.extend(segment_path[:-1])
            else:
                full_path.extend(segment_path)  # Для последнего сегмента включаем конечную точку

            # Рассчитываем стоимость сегмента
            segment_length = get_path_length_from_cache_or_calculate(current_path[i], current_path[i + 1],
                                                                     _terrain_map, path_cache, path_length_cache)
            total_cost += segment_length

        # Если найден новый минимальный путь, сохраняем его
        if total_cost < min_cost:
            min_cost = total_cost
            final_path = full_path

    # Выводим количество шагов
    print(f"Количество перестановок (маршрутов): {total_permutations}")
    print(f"Количество проверок путей: {total_path_checks}")
    print(f"Количество реальных вычислений путей: {len(path_length_cache)}")

    # Очищаем кэш путей после завершения маршрута
    path_cache.clear()
    path_length_cache.clear()
    return final_path
