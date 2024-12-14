from map import generate_map, prepare_basic_map, generate_random_points, visualize_route
from neighbor import nearest_neighbor_routing
from genetic import genetic_algorithm_routing
from brute_force import brute_force_routing
from annealing import simulated_annealing_routing
from paths import a_star_path, calculate_path_length
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from settings import *

# Задание точек маршрута
points: List[Tuple[int, int]]

if choose_points == 0:
    points = generate_random_points((0, 0), (512, 512), num_random_points)
elif choose_points == 8:
    points = [(33, 273), (174, 173), (269, 12), (386, 477), (222, 454), (245, 152), (42, 337), (354, 279)]
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
    path_a_star_forward: List[Tuple[int, int]] = a_star_path(start, end, terrain_map, show_visited_nodes)
    path_length = round(calculate_path_length(path_a_star_forward, terrain_map), 5)
    visualize_route(path_a_star_forward, path_length, "A* start - end", start, [], end, terrain_map)

if do_a_star_backward:
    # Поиск маршрута от конца до начала
    path_a_star_backward: List[Tuple[int, int]] = a_star_path(end, start, terrain_map, show_visited_nodes)
    path_length = round(calculate_path_length(path_a_star_backward, terrain_map), 5)

    visualize_route(path_a_star_backward, path_length, "A* end - start", end, [], start, terrain_map)

if do_brute_force:
    # Поиск маршрута методом грубой силы
    path_brute_force: List[Tuple[int, int]] = brute_force_routing(start, points, end, terrain_map)
    path_length = round(calculate_path_length(path_brute_force, terrain_map), 5)

    visualize_route(path_brute_force, path_length, "Алгоритм грубой силы", start, points, end, terrain_map)

if do_nearest_neighbor:
    # Поиск маршрута методом ближайшего соседа
    path_nearest_neighbor: List[Tuple[int, int]] = nearest_neighbor_routing(start, points, end, terrain_map)
    path_length = round(calculate_path_length(path_nearest_neighbor, terrain_map), 5)

    visualize_route(path_nearest_neighbor, path_length, "Алгоритм ближайшего соседа", start, points, end, terrain_map)

if do_annealing:
    # Поиск маршрута методом симулированного отжига
    path_annealing: List[Tuple[int, int]] = simulated_annealing_routing(start, points, end, terrain_map, initial_temp,
                                                                        cooling_rate, iterations)
    path_length = round(calculate_path_length(path_annealing, terrain_map), 5)

    visualize_route(path_annealing, path_length, "Алгоритм симулированного отжига", start, points, end, terrain_map)

if do_genetic:
    # Генетический поиск маршрута
    path_genetic: List[Tuple[int, int]] = genetic_algorithm_routing(start, points, end, terrain_map,
                                                                    population_size, generations,
                                                                    mutation_rate, tournament_size)
    path_length = round(calculate_path_length(path_genetic, terrain_map), 5)

    visualize_route(path_genetic, path_length, "Генетический алгоритм", start, points, end, terrain_map)
