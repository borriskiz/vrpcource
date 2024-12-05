import numpy as np
import noise
import heapq
import matplotlib.pyplot as plt
from typing import List, Tuple
import random

# Параметры карты
width: int = 512
height: int = 512
scale: float = 100.0  # Масштаб шума

# Параметры шума
octaves: int = 8  # Количество октав
persistence: float = 0.5  # Влияние каждой октавы
lacunarity: float = 2.0  # Частота осцилляций

# Параметры маршрута
start: Tuple[int, int] = (100, 100)  # Начальная точка
end: Tuple[int, int] = (400, 400)  # Конечная точка
num_points: int = 10  # Количество дополнительных точек
to_random: bool = False  # Генерировать ли случайные числа

# Параметры генерационного метода
population_size = 50  # Размер популяции
generations = 100  # Количество поколений
mutation_rate = 0.5  # Вероятность мутации
tournament_size = 5  # Количество агентов для отбора


# Функция для генерации случайных точек в пределах определенной области
def generate_random_points(_start: Tuple[int, int], _end: Tuple[int, int], _num_points: int, margin: int = 10) -> List[
    Tuple[int, int]]:
    _points = []
    # Определяем границы, чтобы точки не были слишком близко к краям
    x_min = min(_start[0], _end[0]) + margin
    x_max = max(_start[0], _end[0]) - margin
    y_min = min(_start[1], _end[1]) + margin
    y_max = max(_start[1], _end[1]) - margin

    while len(_points) < _num_points:
        # Генерируем случайные точки в пределах границ
        new_point = (random.randint(x_min, x_max), random.randint(y_min, y_max))

        # Проверяем, что новая точка не слишком близка к уже существующим точкам
        if not any(abs(new_point[0] - p[0]) < 10 and abs(new_point[1] - p[1]) < 10 for p in _points):
            _points.append(new_point)

    return _points


# Задание точек маршрута
if to_random:
    points: List[Tuple[int, int]] = generate_random_points((0, 0), (512, 512), num_points)
else:
    points: List[Tuple[int, int]] = [(300, 157), (72, 36), (367, 366), (42, 337), (354, 279), (489, 307), (476, 344),
                                     (224, 231), (157, 187), (369, 457)]

# Выводим результат
print("Точки маршрута:", points)


# Генерация карты
def generate_map(_width: int, _height: int, _scale: float, _octaves: int, _persistence: float,
                 _lacunarity: float) -> np.ndarray:
    map_data: np.ndarray = np.zeros((_height, _width))
    for y in range(_height):
        for x in range(_width):
            map_data[y][x] = noise.pnoise2(
                x / _scale,
                y / _scale,
                octaves=_octaves,
                persistence=_persistence,
                lacunarity=_lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=42
            )
    return map_data


# Класс для узлов в графе
class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float, h_cost: float, parent: 'Node' = None) -> None:
        self.position: Tuple[int, int] = position
        self.g_cost: float = g_cost
        self.h_cost: float = h_cost
        self.f_cost: float = g_cost + h_cost
        self.parent: 'Node' = parent

    def __lt__(self, other: 'Node') -> bool:
        return self.f_cost < other.f_cost


# Эвристическая функция
def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Функция для вычисления длины маршрута с учетом ландшафта
def calculate_path_length(path: List[Tuple[int, int]], _terrain_map: np.ndarray) -> float:
    length = 0
    for j in range(len(path) - 1):
        x1, y1 = path[j]
        x2, y2 = path[j + 1]
        terrain_cost = abs(_terrain_map[x2, y2] - _terrain_map[x1, y1])
        length += heuristic((x1, y1), (x2, y2)) + terrain_cost
    return length


# Поиск пути с использованием A*
def a_star(_start: Tuple[int, int], _end: Tuple[int, int], _terrain_map: np.ndarray) -> List[Tuple[int, int]]:
    open_list: List[Node] = []
    closed_list: set[Tuple[int, int]] = set()

    start_node: Node = Node(_start, 0, heuristic(_start, _end))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node: Node = heapq.heappop(open_list)
        if current_node.position == _end:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_list.add(current_node.position)

        for neighbor in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor_pos = (current_node.position[0] + neighbor[0], current_node.position[1] + neighbor[1])
            if 0 <= neighbor_pos[0] < _terrain_map.shape[0] and 0 <= neighbor_pos[1] < _terrain_map.shape[1]:
                if neighbor_pos in closed_list:
                    continue

                g_cost = current_node.g_cost + abs(
                    float(_terrain_map[neighbor_pos[0], neighbor_pos[1]]) -
                    float(_terrain_map[current_node.position[0], current_node.position[1]])
                )
                h_cost = heuristic(neighbor_pos, _end)
                neighbor_node = Node(neighbor_pos, g_cost, h_cost, current_node)
                heapq.heappush(open_list, neighbor_node)
    return []


# Функция маршрутизации методом ближайшего соседа
def nearest_neighbor_routing(_start: Tuple[int, int], _points: List[Tuple[int, int]], _end: Tuple[int, int],
                             _terrain_map: np.ndarray) -> List[Tuple[int, int]]:
    unvisited_points = _points.copy()
    current_point = _start
    total_path = [_start]

    while unvisited_points:
        closest_point = min(unvisited_points, key=lambda p: heuristic(current_point, p))
        path_segment = a_star(current_point, closest_point, _terrain_map)
        total_path.extend(path_segment[1:])
        current_point = closest_point
        unvisited_points.remove(closest_point)

    # Добавляем путь к конечной точке
    final_path_segment = a_star(current_point, _end, _terrain_map)
    total_path.extend(final_path_segment[1:])
    return total_path


def prepare_basic_map() -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(terrain_map, cmap='terrain')
    plt.colorbar(label="Высота")
    plt.scatter([p[1] for p in points], [p[0] for p in points], color="purple", s=50, label="Точки маршрута")
    plt.scatter(start[1], start[0], color="green", s=80, label="Начальная точка")
    plt.scatter(end[1], end[0], color="red", s=80, label="Конечная точка")


def draw_path(_path: List[Tuple[int, int]]) -> None:
    for i in range(len(_path) - 1):
        plt.plot([_path[i][1], _path[i + 1][1]],
                 [_path[i][0], _path[i + 1][0]], color="red", linewidth=1, zorder=2)
        if i % 30 == 0:  # Рисуем стрелки каждые 30 шагов
            plt.arrow(
                _path[i][1], _path[i][0],
                _path[i + 1][1] - _path[i][1],
                _path[i + 1][0] - _path[i][0],
                head_width=5, head_length=8, fc="blue", ec="blue", zorder=3
            )


# Генерация карты
terrain_map: np.ndarray = generate_map(width, height, scale, octaves, persistence, lacunarity)

# Визуализация пути
prepare_basic_map()

plt.title("Оригинальный ландшафт")
plt.legend()
plt.show()

# Поиск маршрута методом ближайшего соседа
path_nearest_neighbor: List[Tuple[int, int]] = nearest_neighbor_routing(start, points, end, terrain_map)

# Выводим длину пути
print(f"Длина пути (алгоритм ближайшего соседа): {calculate_path_length(path_nearest_neighbor, terrain_map)}")

# Визуализация пути
prepare_basic_map()
draw_path(path_nearest_neighbor)

plt.title("Алгоритм ближайшего соседа")
plt.legend()
plt.show()

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


# Мутация - инвертирование подотрезка
def reverse_mutation(route: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    mutated_route = route.copy()
    idx1, idx2 = sorted(random.sample(range(len(route)), 2))
    mutated_route[idx1:idx2 + 1] = reversed(mutated_route[idx1:idx2 + 1])
    return mutated_route


# Мутация - перестановка двух случайных элементов
def swap_mutation(route: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    mutated_route = route.copy()
    idx1, idx2 = random.sample(range(len(route)), 2)
    mutated_route[idx1], mutated_route[idx2] = mutated_route[idx2], mutated_route[idx1]
    return mutated_route


# Мутация - случайное изменение маршрута
def random_mutation(route: List[Tuple[int, int]], _points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    mutated_route = route.copy()
    mutation_point = random.randint(0, len(route) - 1)
    mutated_route[mutation_point] = random.choice(_points)  # Заменяем одну точку на случайную
    return mutated_route


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


# Турнирный отбор с агрессивным отбором лучших особей
def tournament_selection(population: List[List[Tuple[int, int]]], _terrain_map: np.ndarray,
                         _tournament_size: int = 5) -> \
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

    for generation in range(_generations):
        # Оценка маршрутов
        evaluated_population = []
        for route in population:
            full_route = [_start] + route + [_end]

            # Проверка валидности маршрута
            if not check_valid_route(full_route):
                continue  # Если маршрут невалиден, пропускаем его

            path = [a_star(full_route[j], full_route[j + 1], _terrain_map) for j in range(len(full_route) - 1)]
            path = [node for segment in path if segment for node in segment[1:]]
            evaluated_population.append((route, calculate_path_length(path, _terrain_map)))

        # Сортировка по качеству (длина маршрута)
        evaluated_population.sort(key=lambda x: x[1])
        best_routes = [route for route, _ in evaluated_population[:_population_size // 2]]

        # Генерация нового поколения с элитизмом
        new_population = best_routes[:]

        # Механизм элитарности: сохраняем лучших особей и создаем новые маршруты
        while len(new_population) < _population_size:
            parent1 = tournament_selection(best_routes, _terrain_map, _tournament_size)
            parent2 = tournament_selection(best_routes, _terrain_map, _tournament_size)

            # Скрещивание с использованием Order Crossover
            child = order_crossover(parent1, parent2)

            # Мутация
            if random.random() < _mutation_rate:
                if random.random() < 0.33:
                    child = reverse_mutation(child)
                elif random.random() < 0.66:
                    child = swap_mutation(child)
                else:
                    child = random_mutation(child, _points)

            # Проверка на валидность после мутации
            if check_valid_route(child):
                new_population.append(child)

        population = new_population

        # Вывод прогресса
        if generation % 10 == 0:
            best_length = calculate_path_length([_start] + best_routes[0] + [_end], _terrain_map)
            print(f"Поколение {generation}, лучший маршрут длина: {best_length}")

    # Лучший маршрут
    best_route = population[0]
    full_route = [_start] + best_route + [_end]

    # Проверка валидности финального маршрута
    if not check_valid_route(full_route):
        raise ValueError("Финальный маршрут содержит некорректные данные.")

    path = [a_star(full_route[j], full_route[j + 1], _terrain_map) for j in range(len(full_route) - 1)]
    final_path = [node for segment in path if segment for node in segment[1:]]
    return final_path


# Генетический поиск маршрута
path_genetic_algorithm: List[Tuple[int, int]] = genetic_algorithm_routing(start, points, end, terrain_map,
                                                                          population_size, generations,
                                                                          mutation_rate, tournament_size)

# Визуализация генетического алгоритма
prepare_basic_map()
draw_path(path_genetic_algorithm)

plt.title("Генетический алгоритм")
plt.legend(loc="best")
plt.show()
