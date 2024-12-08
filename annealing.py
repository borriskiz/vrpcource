import random
import math
from typing import List, Tuple
import numpy as np
from paths import get_path_from_cache_or_calculate, calculate_path_length

path_cache = {}


# Функция для генерации соседнего маршрута (изменение порядка промежуточных точек)
def generate_neighbor(route: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    new_route = route.copy()
    # Генерируем два случайных индекса для промежуточных точек (не включая начальную и конечную точку)
    idx1, idx2 = random.sample(range(1, len(route) - 1), 2)
    new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]  # Меняем местами их
    return new_route


# Функция для расчета длины пути с использованием get_path_from_cache_or_calculate
def calculate_total_cost(route: List[Tuple[int, int]], _terrain_map: np.ndarray) -> float:
    total_path = []

    # Рассчитываем путь от начальной точки до первой промежуточной
    for i in range(len(route) - 1):
        segment = get_path_from_cache_or_calculate(route[i], route[i + 1], _terrain_map, path_cache)
        total_path.extend(segment[1:])  # Исключаем повторение начальной точки сегмента

    # Подсчитываем стоимость пути
    return calculate_path_length(total_path, _terrain_map)


# Функция для симулированного отжига
def simulated_annealing_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                                _terrain_map: np.ndarray, _initial_temp: float, _cooling_rate: float,
                                _iterations: int) -> List[Tuple[int, int]]:
    # Инициализация начального решения: стартовая точка + промежуточные точки + конечная точка
    current_solution = [_start] + _points + [_end]
    current_cost = calculate_total_cost(current_solution, _terrain_map)

    best_solution = current_solution
    best_cost = current_cost

    # Начальная температура
    temperature = _initial_temp

    for iteration in range(_iterations):
        # Генерация соседнего решения (меняем местами промежуточные точки)
        neighbor_solution = generate_neighbor(current_solution)  # Теперь генерируем соседей со всеми точками
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

        # Понижение температуры
        temperature *= _cooling_rate

        # Если температура слишком мала, прекращаем итерации
        if temperature < 1e-5:
            break

        # Понижаем скорость охлаждения на каждом шаге (вы можете настроить)
        if iteration % 100 == 0:
            print(
                f"Итерация {iteration}, Текущая стоимость: {current_cost}, Лучшее решение: {best_cost}, Температура: {temperature}")

    # Возвращаем лучшее найденное решение
    # Здесь обязательно добавляем промежуточные точки в итоговый путь
    final_path = [_start]  # Добавляем начальную точку в путь
    for i in range(len(best_solution) - 1):
        segment = get_path_from_cache_or_calculate(best_solution[i], best_solution[i + 1], _terrain_map, path_cache)
        final_path.extend(segment[1:])  # добавляем без начальной точки сегмента
    # print(f"Итоговый путь: {final_path}")
    return final_path
