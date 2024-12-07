from typing import List, Tuple
import random
import numpy as np
from paths import a_star_path, calculate_path_length


# Мутация - инвертирование подотрезка
def inverse_mutation(route: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    mutated_route = route.copy()
    idx1, idx2 = sorted(random.sample(range(len(route)), 2))
    mutated_route[idx1:idx2 + 1] = reversed(mutated_route[idx1:idx2 + 1])
    return mutated_route


# Мутация - перемешивание подотрезка
def scramble_mutation(route: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    mutated_route = route.copy()
    idx1, idx2 = sorted(random.sample(range(len(route)), 2))

    sub_route = mutated_route[idx1:idx2 + 1]
    random.shuffle(sub_route)
    mutated_route[idx1:idx2 + 1] = sub_route

    return mutated_route


# Мутация - перестановка двух случайных элементов
def swap_mutation(route: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    mutated_route = route.copy()
    idx1, idx2 = random.sample(range(len(route)), 2)
    mutated_route[idx1], mutated_route[idx2] = mutated_route[idx2], mutated_route[idx1]
    return mutated_route


# Диверсификация мутаций
def diversified_mutation(_route: List[Tuple[int, int]], _generation: int, _max_generations: int) -> List[
    Tuple[int, int]]:
    mutation_choice = random.random()
    if _generation < _max_generations * 0.5:  # Для начальных поколений более случайные мутации
        if mutation_choice < 0.33:
            return inverse_mutation(_route)
        elif mutation_choice < 0.66:
            return scramble_mutation(_route)
        else:
            return swap_mutation(_route)
    else:  # Для поздних поколений более "серьезные" мутации
        if mutation_choice < 0.5:
            return inverse_mutation(_route)
        else:
            return scramble_mutation(_route)


def order_crossover(parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # Размер маршрута
    size = len(parent1)

    # Выбираем случайный отрезок для обмена
    _start, _end = sorted(random.sample(range(size), 2))

    # Сохраняем часть маршрута из первого родителя
    child = [(-1, -1)] * size  # Маркер -1, который будет использоваться для пустых мест
    child[_start:_end + 1] = parent1[_start:_end + 1]

    # Заполняем оставшиеся позиции второго родителя
    current_position = (_end + 1) % size
    for point in parent2:
        if point not in child:
            # Ищем первое свободное место
            while child[current_position] != (-1, -1):
                current_position = (current_position + 1) % size
            child[current_position] = point

    # Проверка, что все позиции заполнены
    if (-1, -1) in child:
        raise ValueError("Не все позиции заполнены корректно.")

    return child


# Partially Mapped Crossover (PMX)
def pmx_crossover(parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))  # Случайный диапазон для обмена
    child = [(-1, -1)] * size  # Инициализируем пустой маршрут, маркером (-1, -1)

    # Копируем сегмент из первого родителя в новый маршрут
    for i in range(start, end + 1):
        child[i] = parent1[i]

    # Вставляем недостающие элементы из второго родителя
    for i in range(size):
        if child[i] == (-1, -1):  # Если позиция ещё пустая
            # Ищем элемент из второго родителя, который ещё не вставлен в ребёнка
            for j in range(size):
                if parent2[j] not in child:
                    child[i] = parent2[j]
                    break
    # Проверка, что все позиции заполнены
    if (-1, -1) in child:
        raise ValueError("Не все позиции заполнены корректно.")
    return child


# Турнирный отбор с агрессивным отбором лучших особей
def tournament_selection(population: List[List[Tuple[int, int]]], _terrain_map: np.ndarray,
                         _tournament_size: int) -> \
        List[Tuple[int, int]]:
    tournament = random.sample(population, _tournament_size)
    evaluated_tournament = [(route, calculate_path_length(route, _terrain_map)) for route in tournament]
    evaluated_tournament.sort(key=lambda x: x[1])  # Сортируем по длине пути

    # Проверка, что лучший маршрут корректно выбран
    best_route = evaluated_tournament[0][0]
    assert best_route is not None, "Лучший маршрут не может быть пустым"
    assert len(best_route) == len(population[
                                      0]), f"Размер маршрута несоответствует ожиданиям. Ожидалось {len(population[0])}, но получено {len(best_route)}"
    assert check_valid_route(best_route), "Лучший маршрут невалиден."

    return best_route


# Адаптивный размер турнира
def adaptive_tournament_selection(population: List[List[Tuple[int, int]]], _terrain_map: np.ndarray,
                                  generation: int, _generations: int, _tournament_base_size: int) -> List[
    Tuple[int, int]]:
    # Увеличиваем размер турнира на поздних этапах
    tournament_size = max(_tournament_base_size, int(_tournament_base_size * (1 + generation / _generations)))
    return tournament_selection(population, _terrain_map, tournament_size)


# Проверка, что передаваемые точки не являются None
def check_valid_route(route: List[Tuple[int, int]]) -> bool:
    # Проверка на наличие None или маршрута длины 1
    if None in route or (-1, -1) in route or len(route) < 2:
        return False
    # Также можно добавить проверку на дублирующиеся точки
    return len(set(route)) == len(route)


# Генетический алгоритм
def genetic_algorithm_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                              _terrain_map: np.ndarray, _population_size: int, _generations: int,
                              _mutation_rate: float, _tournament_size: int) -> List[Tuple[int, int]]:
    # Инициализация популяции
    population = [random.sample(_points, len(_points)) for _ in range(_population_size)]

    best_solution = None
    best_length = float('inf')

    for generation in range(_generations):
        # Оценка маршрутов
        evaluated_population = []
        for route in population:
            full_route = [_start] + route + [_end]

            if not check_valid_route(full_route):
                continue  # Пропускаем невалидные маршруты

            path = [a_star_path(full_route[j], full_route[j + 1], _terrain_map) for j in range(len(full_route) - 1)]
            path = [node for segment in path if segment for node in segment[1:]]
            length = calculate_path_length(path, _terrain_map)
            evaluated_population.append((route, length))

            # Обновляем лучший маршрут, если найден более короткий
            if length < best_length:
                best_length = length
                best_solution = full_route

        # Сортировка по длине пути
        evaluated_population.sort(key=lambda x: x[1])
        best_routes = [route for route, _ in evaluated_population[:_population_size // 2]]

        # Новый набор популяции с элитизмом
        new_population = best_routes[:]

        while len(new_population) < _population_size:
            # parent1 = tournament_selection(best_routes, _terrain_map, _tournament_size)
            # parent2 = tournament_selection(best_routes, _terrain_map, _tournament_size)
            parent1 = adaptive_tournament_selection(best_routes, _terrain_map, generation, _generations,
                                                    _tournament_size)
            parent2 = adaptive_tournament_selection(best_routes, _terrain_map, generation, _generations,
                                                    _tournament_size)

            # child = order_crossover(parent1, parent2)
            child = pmx_crossover(parent1, parent2)

            if random.random() < _mutation_rate:
                # Применяем диверсифицированную мутацию
                child = diversified_mutation(child, generation, _generations)

            if check_valid_route(child):
                new_population.append(child)

        population = new_population

        # Печать прогресса
        if generation % 10 == 0:
            print(f"Поколение {generation}, лучший длина лучшего маршрута: {best_length}")

    # Возвращаем лучший найденный маршрут
    if best_solution is None:
        raise ValueError("Не найдено корректного маршрута.")

    path = [a_star_path(best_solution[j], best_solution[j + 1], _terrain_map) for j in range(len(best_solution) - 1)]
    final_path = [node for segment in path if segment for node in segment[1:]]
    return final_path
