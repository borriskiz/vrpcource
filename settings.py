from typing import Tuple

# Параметры выполняемых тестовых опытов
do_original: bool = False
show_visited_nodes: bool = True
do_a_star_forward: bool = False
do_a_star_backward: bool = False

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
choose_points: int = 20  # 0 Генерировать ли случайные числа, 8 чисел, 10 чисел, 20 чисел

# Параметры генетического метода
population_size: int = 100  # Размер популяции
generations: int = 100  # Количество поколений
mutation_rate: float = 0.3  # Вероятность мутации
tournament_size: int = population_size // 10  # Количество агентов для отбора
output_gene_crossover: bool = False
output_gene_mutation: bool = False
plot_graph_genetic: bool = True

# Параметры симулированного отжига
initial_temp: float = 2000
cooling_rate: float = 0.995
iterations: int = 1000
plot_graph_annealing: bool = True
