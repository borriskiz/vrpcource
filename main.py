from map import generate_map, prepare_basic_map, draw_path, generate_random_points
from neighbor import nearest_neighbor_routing
from genetic import genetic_algorithm_routing
from brute_force import brute_force_routing
from annealing import simulated_annealing_routing
from paths import calculate_path_length, a_star_path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Параметры выполняемых тестовых опытов
do_original: bool = False
do_a_star_forward: bool = True
do_a_star_backward: bool = True

# Параметры выполняемых опытов
do_brute_force: bool = False
do_nearest_neighbor: bool = True
do_annealing: bool = True
do_genetic: bool = True

# Параметры карты
width: int = 512
height: int = 512
scale: float = 100.0  # Масштаб шума

# Параметры шума
octaves: int = 10  # Количество октав
persistence: float = 0.5  # Влияние каждой октавы
lacunarity: float = 2.0  # Частота осцилляций

# Параметры маршрута
start: Tuple[int, int] = (100, 100)  # Начальная точка
end: Tuple[int, int] = (400, 400)  # Конечная точка
num_random_points: int = 20  # Количество случайных точек
choose_points: int = 20  # 0 Генерировать ли случайные числа, 6 чисел, 10 чисел, 20 чисел

# Параметры генетического метода
population_size: int = 100  # Размер популяции
generations: int = 100  # Количество поколений
mutation_rate: float = 0.3  # Вероятность мутации
tournament_size: int = population_size // 10  # Количество агентов для отбора

# Параметры симулированного отжига
initial_temp: float = 1000
cooling_rate: float = 0.998
iterations: int = 1000


def visualize_route(_path: List[Tuple[int, int]], _algorithm_name: str, _start: Tuple[int, int],
                    _points: List[Tuple[int, int]], _end: Tuple[int, int], _terrain_map: np.ndarray):
    # Расчет длины пути и округление
    path_length = round(calculate_path_length(_path, _terrain_map), 3)
    print(f"Длина пути ({_algorithm_name}): {path_length}\n")

    # Визуализация пути
    prepare_basic_map(_start, _end, _points, _terrain_map)
    draw_path(_path)

    # Заголовок с длиной пути
    plt.title(f"{_algorithm_name}, Длина пути: {path_length}")
    plt.legend()
    plt.show()


# Задание точек маршрута
points: List[Tuple[int, int]]

if choose_points == 0:
    points = generate_random_points((0, 0), (512, 512), num_random_points)
elif choose_points == 6:
    points = [(33, 273), (174, 173), (269, 12), (386, 477), (222, 454), (245, 152)]
elif choose_points == 10:
    points = [(300, 157), (72, 36), (367, 366), (42, 337), (354, 279), (489, 307), (476, 344),
              (224, 231), (157, 187), (369, 457)]
elif choose_points == 20:
    points = [(139, 113), (195, 252), (21, 494), (45, 407), (43, 280), (49, 340), (338, 31),
              (353, 149), (173, 13), (271, 139), (23, 274), (218, 395), (490, 110), (369, 379),
              (367, 401), (79, 250), (214, 109), (245, 412), (191, 311), (214, 443)]
else:
    points = generate_random_points((0, 0), (512, 512), num_random_points)

# Выводим результат
print("Точки маршрута:", points)

# Генерация карты
terrain_map: np.ndarray = generate_map(width, height, scale, octaves, persistence, lacunarity)
if do_original:
    # Визуализация пути
    prepare_basic_map(start, end, points, terrain_map)
    plt.title("Оригинальный ландшафт")
    plt.legend()
    plt.show()

if do_a_star_forward:
    # Поиск маршрута от начала до конца
    path_a_star_forward: List[Tuple[int, int]] = a_star_path(start, end, terrain_map, True)
    visualize_route(path_a_star_forward, "A* start - end", start, [], end, terrain_map)

if do_a_star_backward:
    # Поиск маршрута от конца до начала
    path_a_star_backward: List[Tuple[int, int]] = a_star_path(end, start, terrain_map, True)
    visualize_route(path_a_star_backward, "A* end - start", end, [], start, terrain_map)

if do_brute_force:
    # Поиск маршрута методом грубой силы
    path_brute_force: List[Tuple[int, int]] = brute_force_routing(start, points, end, terrain_map)
    visualize_route(path_brute_force, "Алгоритм грубой силы", start, points, end, terrain_map)

if do_nearest_neighbor:
    # Поиск маршрута методом ближайшего соседа
    path_nearest_neighbor: List[Tuple[int, int]] = nearest_neighbor_routing(start, points, end, terrain_map)
    visualize_route(path_nearest_neighbor, "Алгоритм ближайшего соседа", start, points, end, terrain_map)

if do_annealing:
    # Поиск маршрута методом симулированного отжига
    path_annealing: List[Tuple[int, int]] = simulated_annealing_routing(start, points, end, terrain_map, initial_temp,
                                                                        cooling_rate, iterations)
    visualize_route(path_annealing, "Алгоритм симулированного отжига", start, points, end, terrain_map)

if do_genetic:
    # Генетический поиск маршрута
    path_genetic: List[Tuple[int, int]] = genetic_algorithm_routing(start, points, end, terrain_map,
                                                                    population_size, generations,
                                                                    mutation_rate, tournament_size)
    visualize_route(path_genetic, "Генетический алгоритм", start, points, end, terrain_map)
