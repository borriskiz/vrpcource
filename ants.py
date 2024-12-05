import random
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from paths import a_star_path, heuristic, calculate_path_length


# Функция для вычисления вероятности перехода между двумя точками
def transition_probability(_pheromone: float, _distance: float, _alpha: float, _beta: float) -> float:
    return (_pheromone ** _alpha) * (1.0 / _distance) ** _beta


# Функция для добавления промежуточных точек между двумя точками с использованием A*
def add_intermediate_points(start: Tuple[int, int], end: Tuple[int, int], terrain_map: np.ndarray) -> List[
    Tuple[int, int]]:
    # Получаем путь с учётом рельефа
    path = a_star_path(start, end, terrain_map)
    return path  # Все точки пути включая начальную и конечную


# Основной алгоритм муравьиной колонии с учетом рельефа
def ant_colony_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                       _terrain_map: np.ndarray, _num_ants: int, _num_iterations: int, _alpha: float,
                       _beta: float, _rho: float, _q: float) -> List[Tuple[int, int]]:
    # Обновляем список точек, добавляя стартовую и конечную точку
    points = _points.copy()
    if _start not in points:
        points.insert(0, _start)
    if _end not in points:
        points.append(_end)

    # Инициализация феромонов (по умолчанию равномерное распределение)
    pheromone_map = defaultdict(lambda: defaultdict(lambda: 1.0))

    best_route = None
    best_length = float('inf')

    for iteration in range(_num_iterations):
        all_routes = []
        all_lengths = []

        for ant in range(_num_ants):
            # Инициализация маршрута муравья
            route = [_start]
            visited = {point for point in route}  # Множество посещённых точек
            current_point = _start

            while len(route) < len(points):  # Пока не все точки посещены
                probabilities = []
                for point in points:
                    if point not in visited:
                        # Используем A* для поиска пути с учетом рельефа
                        path_segment = add_intermediate_points(current_point, point, _terrain_map)
                        dist = sum(
                            heuristic(path_segment[i], path_segment[i + 1]) for i in range(len(path_segment) - 1))

                        # Получаем феромоны для перехода, берем одно значение
                        pheromone = pheromone_map[current_point][point]

                        prob = transition_probability(pheromone, dist, _alpha, _beta)
                        probabilities.append((point, prob))

                # Нормализуем вероятности
                total_prob = sum(prob for _, prob in probabilities)
                probabilities = [(point, prob / total_prob) for point, prob in probabilities]

                # Выбираем точку с максимальной вероятностью
                next_point = random.choices([point for point, _ in probabilities],
                                            [prob for _, prob in probabilities])[0]

                route.append(next_point)
                visited.add(next_point)
                current_point = next_point

            # Завершаем путь до конечной точки с учетом рельефа
            intermediate_points = add_intermediate_points(current_point, _end, _terrain_map)
            route.extend(intermediate_points)
            total_length = calculate_path_length(route, _terrain_map)

            all_routes.append(route)
            all_lengths.append(total_length)

            # Обновляем лучший путь
            if total_length < best_length:
                best_length = total_length
                best_route = route

        # Печать прогресса
        if iteration % 1 == 0:
            print(f"Итерация {iteration}, лучший длина лучшего маршрута: {best_length}")

        # Обновление феромонов
        for i in range(_num_ants):
            for j in range(len(all_routes[i]) - 1):
                start_point = all_routes[i][j]
                end_point = all_routes[i][j + 1]

                # Обновляем феромоны на пути
                pheromone_map[start_point][end_point] += _q / all_lengths[i]

        # Испарение феромонов
        for start in pheromone_map:
            for end in pheromone_map[start]:
                pheromone_map[start][end] *= (1 - _rho)

    return best_route
