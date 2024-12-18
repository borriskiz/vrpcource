from typing import List, Tuple
import random
import numpy as np
from paths import get_path_from_cache_or_calculate, get_path_length_from_cache_or_calculate
from map import plot_generation_data
from settings import output_gene_crossover, output_gene_mutation, plot_graph_genetic

path_cache = {}
path_length_cache = {}

# Счетчик проверок путей
path_check_count: int = 0


# Мутация - инвертирование части отрезка
def inverse_mutation(route: List[int]) -> List[int]:
    if output_gene_mutation:
        print(f"route before inverse mutation: {route}")
    mutated_route = route.copy()
    idx1, idx2 = sorted(random.sample(range(len(route)), 2))
    mutated_route[idx1:idx2 + 1] = reversed(mutated_route[idx1:idx2 + 1])
    if output_gene_mutation:
        print(f"route after inverse mutation: {mutated_route}")
    return mutated_route


# Мутация - перемешивание части отрезка
def scramble_mutation(route: List[int]) -> List[int]:
    if output_gene_mutation:
        print(f"route before scramble mutation: {route}")
    mutated_route = route.copy()
    idx1, idx2 = sorted(random.sample(range(len(route)), 2))

    sub_route = mutated_route[idx1:idx2 + 1]
    random.shuffle(sub_route)
    mutated_route[idx1:idx2 + 1] = sub_route
    if output_gene_mutation:
        print(f"route after scramble mutation: {mutated_route}")
    return mutated_route


# Мутация - перестановка двух случайных элементов
def swap_mutation(route: List[int]) -> List[int]:
    if output_gene_mutation:
        print(f"route before swap mutation: {route}")
    mutated_route = route.copy()
    idx1, idx2 = random.sample(range(len(route)), 2)
    mutated_route[idx1], mutated_route[idx2] = mutated_route[idx2], mutated_route[idx1]
    if output_gene_mutation:
        print(f"route after swap mutation: {mutated_route}")
    return mutated_route


# Диверсификация мутаций
def diversified_mutation(_route: List[int], _generation: int, _max_generations: int) -> List[int]:
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


# Partially Mapped Crossover (PMX)
def pmx_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    if output_gene_crossover:
        print(f"parent1: {parent1}")
        print(f"parent2: {parent2}")

    # Создание двух детей
    child1 = [-1] * size
    child2 = [-1] * size

    # Копирование части отрезка от родителя 1 в child1, и от родителя 2 в child2
    for i in range(start, end + 1):
        child1[i] = parent1[i]
        child2[i] = parent2[i]

    if output_gene_crossover:
        print(f"child1 (after part copy): {child1}")
        print(f"child2 (after part copy): {child2}")

    # Заполнение оставшихся пустых мест в child1
    for i in range(size):
        if child1[i] == -1:
            # Пропускаем элементы, которые уже есть в перекопированной части
            current_element = parent2[i]
            while current_element in child1:  # Пропускаем элементы, которые уже есть в child1
                idx = parent1.index(current_element)  # Ищем, где элемент находится в родителе 2
                current_element = parent2[idx]  # Получаем элемент из второго родителя
            child1[i] = current_element

    # Заполнение оставшихся пустых мест в child2
    for i in range(size):
        if child2[i] == -1:
            current_element = parent1[i]
            while current_element in child2:  # Пропускаем элементы, которые уже есть в child2
                idx = parent2.index(current_element)  # Ищем, где элемент находится в родителе 1
                current_element = parent1[idx]  # Получаем элемент из первого родителя
            child2[i] = current_element

    # Проверка, что оба ребенка корректно заполнились
    if -1 in child1 or -1 in child2:
        raise ValueError("Не все позиции заполнены корректно.")

    if output_gene_crossover:
        print(f"child1 (final): {child1}")
        print(f"child2 (final): {child2}")

    return child1, child2


# Турнирный отбор с агрессивным отбором лучших особей
def tournament_selection(population: List[List[int]], _terrain_map: np.ndarray, _tournament_size: int,
                         points: List[Tuple[int, int]]) -> List[int]:
    tournament = random.sample(population, _tournament_size)

    evaluated_tournament = []
    for route in tournament:
        total_length = 0
        for i in range(len(route) - 1):
            # Подсчитываем количество проверок путей
            global path_check_count
            path_check_count += 1  # Каждая проверка пути увеличивает счетчик
            total_length += get_path_length_from_cache_or_calculate(points[route[i]], points[route[i + 1]],
                                                                    _terrain_map,
                                                                    path_cache, path_length_cache)

        evaluated_tournament.append((route, total_length))

    evaluated_tournament.sort(key=lambda x: x[1])

    best_route = evaluated_tournament[0][0]
    assert best_route is not None, "Лучший маршрут не может быть пустым"
    assert len(best_route) == len(population[0]), f"Размер маршрута несоответствует ожиданиям."
    assert check_valid_route(best_route), "Лучший маршрут невалиден."

    return best_route


# Адаптивный размер турнира
def adaptive_tournament_selection(population: List[List[int]], _terrain_map: np.ndarray,
                                  generation: int, _generations: int, _tournament_base_size: int,
                                  points: List[Tuple[int, int]]) -> List[int]:
    tournament_size = max(_tournament_base_size, int(_tournament_base_size * (1 + generation / _generations)))
    return tournament_selection(population, _terrain_map, tournament_size, points)


# Проверка маршрута
def check_valid_route(route: List[int]) -> bool:
    if None in route or -1 in route or len(route) < 2:
        return False
    return len(set(route)) == len(route)


def genetic_algorithm_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                              _terrain_map: np.ndarray, _population_size: int, _generations: int,
                              _mutation_rate: float, _tournament_size: int) -> List[
    Tuple[int, int]]:
    # Убираем начальную и конечную точку из списка точек
    points = [point for point in _points if point != _start and point != _end]

    # Создание начальной популяции маршрутов с индексами точек (исключая начальную и конечную)
    population = [random.sample(range(len(points)), len(points)) for _ in range(_population_size)]

    best_solution = None
    best_length = float('inf')

    # Списки для хранения данных для графика
    generation_lengths = []  # Список длин путей для каждого поколения
    average_lengths = []  # Список средних длин путей для каждого поколения
    min_lengths = []  # Список минимальных длин путей для каждого поколения

    for generation in range(_generations):
        evaluated_population = []
        generation_total_length = 0  # Для подсчета средней длины в поколении
        generation_min_length = float('inf')  # Для подсчета минимальной длины в поколении

        for route in population:
            # Вставляем начальную и конечную точку в начало и конец маршрута
            full_route = [_start] + [points[i] for i in route] + [_end]

            final_path = []
            for j in range(len(full_route) - 1):
                segment = get_path_from_cache_or_calculate(full_route[j], full_route[j + 1], _terrain_map, path_cache,
                                                           path_length_cache)
                final_path.extend(segment[:])

            length = 0.0
            for i in range(len(final_path) - 1):
                length += get_path_length_from_cache_or_calculate(final_path[i], final_path[i + 1], _terrain_map,
                                                                  path_cache, path_length_cache)
            evaluated_population.append((route, length))
            generation_total_length += length

            # Обновляем минимальную длину для поколения
            if length < generation_min_length:
                generation_min_length = length

            if length < best_length:
                best_length = length
                best_solution = full_route

        # Сортируем популяцию по длине пути
        evaluated_population.sort(key=lambda x: x[1])
        best_routes = [route for route, _ in evaluated_population[:_population_size // 2]]

        # Сохраняем информацию о длинах для графика
        generation_lengths.append([length for _, length in evaluated_population])
        average_lengths.append(generation_total_length / len(population))
        min_lengths.append(generation_min_length)

        # Создаем новую популяцию
        new_population = best_routes[:]

        # Создание новой популяции с двумя детьми на каждой итерации
        while len(new_population) < _population_size:
            parent1 = adaptive_tournament_selection(best_routes, _terrain_map, generation, _generations,
                                                    _tournament_size, points)
            parent2 = adaptive_tournament_selection(best_routes, _terrain_map, generation, _generations,
                                                    _tournament_size, points)

            # Получаем два ребенка от кроссовера
            child1, child2 = pmx_crossover(parent1, parent2)

            # Применяем мутацию
            if random.random() < _mutation_rate:
                child1 = diversified_mutation(child1, generation, _generations)
                child2 = diversified_mutation(child2, generation, _generations)

            # Добавляем детей в новую популяцию, если они корректны
            if check_valid_route(child1):
                new_population.append(child1)
            if check_valid_route(child2):
                new_population.append(child2)

        population = new_population

        if generation % 10 == 0:
            print(f"Поколение {generation}, длина лучшего маршрута: {best_length}")

    if best_solution is None:
        raise ValueError("Не найдено корректного маршрута.")

    final_path = []
    for j in range(len(best_solution) - 1):
        segment = get_path_from_cache_or_calculate(best_solution[j], best_solution[j + 1], _terrain_map, path_cache,
                                                   path_length_cache)
        final_path.extend(segment[1:])
    final_path = [_start] + final_path  # Добавляем начальную точку в начало маршрута

    print(f"Количество проверок путей: {path_check_count}")
    print(f"Количество реальных вычислений путей: {len(path_length_cache)}")

    path_cache.clear()
    path_length_cache.clear()

    # Построение графика, если plot_graph == True
    if plot_graph_genetic:
        plot_generation_data(generation_lengths, average_lengths, min_lengths, _generations)

    return final_path
