import random
import numpy as np
from typing import List, Tuple


# Функция для расчета расстояния между двумя точками
def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Функция для вычисления вероятности перехода между двумя точками
def transition_probability(_pheromone: float, _distance: float, _alpha: float, _beta: float) -> float:
    return (_pheromone ** _alpha) * (1.0 / _distance) ** _beta


# Алгоритм муравьиной колонии
def ant_colony_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                       _terrain_map: np.ndarray,
                       _num_ants: int, _num_iterations: int, _alpha: float, _beta: float,
                       _rho: float, _q: float) -> List[Tuple[int, int]]:
    # Инициализация феромонов
    pheromone_map = np.ones((len(_points), len(_points))) * 1.0

    best_route = None
    best_length = float('inf')

    for iteration in range(_num_iterations):
        all_routes = []
        all_lengths = []

        for ant in range(_num_ants):
            # Начальный маршрут для муравья
            route = [_start]
            visited = {_start}
            current_point = _start
            total_length = 0

            while len(route) < len(_points) + 1:  # Путь должен включать все точки
                # Для каждой точки, которая еще не посещена, вычисляем вероятности перехода
                probabilities = []
                for i, point in enumerate(_points):
                    if point not in visited:
                        dist = distance(current_point, point)
                        prob = transition_probability(pheromone_map[current_point, i], dist, _alpha, _beta)
                        probabilities.append((point, prob))

                # Переход к следующей точке по вероятности
                total_prob = sum(prob for _, prob in probabilities)
                probabilities = [(point, prob / total_prob) for point, prob in probabilities]

                # Выбираем точку с максимальной вероятностью
                next_point = random.choices([point for point, _ in probabilities],
                                            [prob for _, prob in probabilities])[0]

                route.append(next_point)
                visited.add(next_point)
                total_length += distance(current_point, next_point)
                current_point = next_point

            # Завершаем путь до конечной точки
            total_length += distance(current_point, _end)
            route.append(_end)
            all_routes.append(route)
            all_lengths.append(total_length)

            # Обновляем лучший путь, если найден более короткий
            if total_length < best_length:
                best_length = total_length
                best_route = route
            if iteration % 10 == 0:
                print(f"Итерация {iteration}, лучший длина лучшего маршрута: {best_length}")

        # Обновление феромонов
        pheromone_map *= (1 - _rho)  # Испарение феромонов
        for i in range(_num_ants):
            # Добавляем феромоны на пути, пройденном муравьем
            for j in range(len(all_routes[i]) - 1):
                start_point = all_routes[i][j]
                end_point = all_routes[i][j + 1]
                pheromone_map[start_point, end_point] += _q / all_lengths[i]

    return best_route
