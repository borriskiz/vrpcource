import random
import numpy as np
from typing import List, Tuple
from paths import a_star_path, heuristic


# Функция для вычисления вероятности перехода между двумя точками
def transition_probability(_pheromone: float, _distance: float, _alpha: float, _beta: float) -> float:
    return (_pheromone ** _alpha) * (1.0 / _distance) ** _beta


# Алгоритм муравьиной колонии с учетом рельефа
def ant_colony_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                                    _terrain_map: np.ndarray, _num_ants: int, _num_iterations: int, _alpha: float,
                                    _beta: float, _rho: float, _q: float) -> List[Tuple[int, int]]:
    # Копируем список точек и добавляем стартовую и конечную точки, если их нет
    points = _points.copy()
    if _start not in points:
        points.insert(0, _start)
    if _end not in points:
        points.append(_end)

    # Создаем словарь для точек и их индексов
    point_to_index = {point: i for i, point in enumerate(points)}

    # Инициализация феромонов (матрица размером N x N, где N - количество точек)
    pheromone_map = np.ones((len(points), len(points))) * 1.0

    best_route = None
    best_length = float('inf')

    # Алгоритм муравьиной колонии
    for iteration in range(_num_iterations):
        all_routes = []
        all_lengths = []

        for ant in range(_num_ants):
            # Начальный маршрут для муравья
            route = [_start]
            visited = {point_to_index[_start]}  # Индексированные посещенные точки
            current_point = _start
            total_length = 0

            while len(route) < len(points):  # Путь должен включать все точки
                probabilities = []
                for i, point in enumerate(points):
                    if point not in route:  # Проверяем, посещена ли точка
                        # Используем A* для поиска пути с учетом рельефа
                        path_segment = a_star_path(current_point, point, _terrain_map)
                        dist = sum(
                            heuristic(path_segment[i], path_segment[i + 1]) for i in range(len(path_segment) - 1))

                        # Получаем феромоны для перехода, берем одно значение
                        pheromone = pheromone_map[point_to_index[current_point], point_to_index[point]]

                        # Проверка на тип данных, чтобы избежать ошибки
                        if isinstance(pheromone, np.ndarray):
                            pheromone = pheromone.item()  # Преобразуем в одно значение float

                        prob = transition_probability(pheromone, dist, _alpha, _beta)
                        probabilities.append((point, prob))

                # Нормализуем вероятности
                total_prob = sum(prob for _, prob in probabilities)
                probabilities = [(point, prob / total_prob) for point, prob in probabilities]

                # Выбираем точку с максимальной вероятностью
                next_point = random.choices([point for point, _ in probabilities],
                                            [prob for _, prob in probabilities])[0]

                route.append(next_point)
                visited.add(point_to_index[next_point])
                total_length += heuristic(current_point, next_point)
                current_point = next_point

            # Завершаем путь до конечной точки с учетом рельефа
            path_segment = a_star_path(current_point, _end, _terrain_map)
            total_length += sum(heuristic(path_segment[i], path_segment[i + 1]) for i in range(len(path_segment) - 1))
            route.extend(path_segment[1:])
            all_routes.append(route)
            all_lengths.append(total_length)

            # Обновляем лучший путь, если найден более короткий
            if total_length < best_length:
                best_length = total_length
                best_route = route

        # Печать прогресса
        if iteration % 10 == 0:
            print(f"Итерация {iteration}, лучший длина лучшего маршрута: {best_length}")

        # Обновление феромонов
        pheromone_map *= (1 - _rho)  # Испарение феромонов
        for i in range(_num_ants):
            # Добавляем феромоны на пути, пройденном муравьем
            for j in range(len(all_routes[i]) - 1):
                start_point = all_routes[i][j]
                end_point = all_routes[i][j + 1]
                pheromone_map[point_to_index[start_point], point_to_index[end_point]] += _q / all_lengths[i]

    return best_route
