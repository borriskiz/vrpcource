import random
import math
from typing import List, Tuple
import numpy as np
from paths import get_path_from_cache_or_calculate, get_path_length_from_cache_or_calculate

path_cache = {}
path_length_cache = {}

# Счетчики для подсчета проверок путей
path_check_count = 0


# Функция для генерации соседнего маршрута (изменение порядка промежуточных точек)
def generate_neighbor(route: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    new_route = route.copy()
    # Генерируем два случайных индекса для промежуточных точек (не включая начальную и конечную точку)
    idx1, idx2 = random.sample(range(1, len(route) - 1), 2)
    new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]  # Меняем местами их
    return new_route


# Функция для расчета длины пути с использованием кэша
def calculate_total_cost(route: List[Tuple[int, int]], _terrain_map: np.ndarray) -> float:
    global path_check_count

    total_path = []
    total_cost = 0.0

    # Рассчитываем путь от начальной точки до первой промежуточной
    for i in range(len(route) - 1):
        # Проверяем, есть ли путь в кэше
        path_check_count += 1  # Каждая проверка пути увеличивает счетчик
        segment = get_path_from_cache_or_calculate(route[i], route[i + 1], _terrain_map, path_cache, path_length_cache)

        total_path.extend(segment)  # Добавляем все точки сегмента, включая начальную

        # Рассчитываем стоимость пути
        total_cost += get_path_length_from_cache_or_calculate(route[i], route[i + 1], _terrain_map, path_cache,
                                                              path_length_cache)

    return total_cost


# Функция для симулированного отжига
def simulated_annealing_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                                _terrain_map: np.ndarray, _initial_temp: float, _cooling_rate: float,
                                _iterations: int) -> List[Tuple[int, int]]:
    global path_check_count

    # Инициализация начального решения: стартовая точка + промежуточные точки + конечная точка
    current_solution = [_start] + _points + [_end]
    current_cost = calculate_total_cost(current_solution, _terrain_map)

    best_solution = current_solution
    best_cost = current_cost

    # Начальная температура
    temperature = _initial_temp

    for iteration in range(_iterations):
        # Генерация соседнего решения (меняем местами промежуточные точки)
        neighbor_solution = generate_neighbor(current_solution)  # Генерация соседа
        # Расчет стоимости нового маршрута
        neighbor_cost = calculate_total_cost(neighbor_solution, _terrain_map)

        # Вычисление изменения стоимости
        delta_cost = neighbor_cost - current_cost

        # Если новое решение лучше, принимаем его
        if delta_cost < 0.0:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
        else:
            # Если новое решение хуже, принимаем его с вероятностью exp(-delta_cost / temperature)
            acceptance_probability = math.exp(-delta_cost / temperature)
            if random.random() < acceptance_probability:
                current_solution = neighbor_solution
                current_cost = neighbor_cost

        # Обновление лучшего решения, если новое лучше
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost

        # Понижаем скорость охлаждения на каждом шаге
        if iteration % 100 == 0:
            print(
                f"Итерация {iteration}, Текущая стоимость: {current_cost}, Лучшее решение: {best_cost}, Температура: {temperature}")

        # Понижение температуры
        temperature *= _cooling_rate

        # Если температура слишком мала, прекращаем итерации
        if temperature < 1e-5:
            break

    final_path = [_start]  # Начальная точка
    for i in range(len(best_solution) - 1):
        segment = get_path_from_cache_or_calculate(best_solution[i], best_solution[i + 1], _terrain_map, path_cache,
                                                   path_length_cache)
        final_path.extend(segment)

    # Выводим количество просчитанных путей и проверок
    print(f"Количество проверок путей: {path_check_count}")
    print(f"Количество реальных вычислений путей: {len(path_length_cache)}")

    # Очищаем кэш путей после завершения маршрута
    path_cache.clear()
    path_length_cache.clear()

    return final_path
