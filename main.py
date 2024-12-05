from map import generate_map, prepare_basic_map, draw_path, generate_random_points
from neighbor import nearest_neighbor_routing
from genetic import genetic_algorithm_routing
from paths import calculate_path_length
from ants import ant_colony_routing
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

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
num_random_points: int = 10  # Количество случайных точек
choose_points: int = 2  # 0 Генерировать ли случайные числа, 1 10 чисел, 2 20 чисел

# Параметры генерационного метода
population_size: int = 100  # Размер популяции
generations: int = 100  # Количество поколений
mutation_rate: float = 0.2  # Вероятность мутации
tournament_size: int = population_size // 10  # Количество агентов для отбора

# Параметры метода муравьиной колонии

alpha = 1  # Влияние феромона
beta = 2  # Влияние эвристики (расстояние)
rho = 0.1  # Коэффициент испарения феромонов
q = 100  # Количество феромонов, оставляемое за успешный путь
num_ants: int = 10  # Количество муравьев
num_iterations: int = 100  # Количество итераций

# Задание точек маршрута
points: List[Tuple[int, int]]

if choose_points == 0:
    points = generate_random_points((0, 0), (512, 512), num_random_points)
elif choose_points == 1:
    points = [(300, 157), (72, 36), (367, 366), (42, 337), (354, 279), (489, 307), (476, 344),
              (224, 231), (157, 187), (369, 457)]
elif choose_points == 2:
    points = [(139, 113), (195, 252), (21, 494), (45, 407), (43, 280), (49, 340), (338, 31),
              (353, 149), (173, 13), (271, 139), (23, 274), (218, 395), (490, 110), (369, 379),
              (367, 401), (79, 250), (214, 109), (245, 412), (191, 311), (214, 443)]
else:
    points = generate_random_points((0, 0), (512, 512), num_random_points)

# Выводим результат
print("Точки маршрута:", points)

# Генерация карты
terrain_map: np.ndarray = generate_map(width, height, scale, octaves, persistence, lacunarity)
# Визуализация пути
prepare_basic_map(start, end, points, terrain_map)
plt.title("Оригинальный ландшафт")
plt.legend()
plt.show()

# Поиск маршрута методом ближайшего соседа
path_nearest_neighbor: List[Tuple[int, int]] = nearest_neighbor_routing(start, points, end, terrain_map)

# Выводим длину пути
print(f"Длина пути (алгоритм ближайшего соседа): {calculate_path_length(path_nearest_neighbor, terrain_map)}")

# Визуализация пути
prepare_basic_map(start, end, points, terrain_map)
draw_path(path_nearest_neighbor)

plt.title("Алгоритм ближайшего соседа")
plt.legend()
plt.show()

# Поиск маршрута методом муравьиной колонии
path_ants: List[Tuple[int, int]] = ant_colony_routing(start, points, end, terrain_map, num_ants, num_iterations, alpha,
                                                      beta, rho, q)

# Выводим длину пути
print(f"Длина пути (алгоритм муравьиной колонии): {calculate_path_length(path_ants, terrain_map)}")

# Визуализация пути
prepare_basic_map(start, end, points, terrain_map)
draw_path(path_ants)

plt.title("Алгоритм муравьиной колонии")
plt.legend()
plt.show()

# Генетический поиск маршрута
path_genetic_algorithm: List[Tuple[int, int]] = genetic_algorithm_routing(start, points, end, terrain_map,
                                                                          population_size, generations,
                                                                          mutation_rate, tournament_size)
# Выводим длину пути
print(f"Длина пути (генетический алгоритм): {calculate_path_length(path_genetic_algorithm, terrain_map)}")

# Визуализация генетического алгоритма
prepare_basic_map(start, end, points, terrain_map)
draw_path(path_genetic_algorithm)

plt.title("Генетический алгоритм")
plt.legend()
plt.show()
