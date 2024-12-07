from typing import Tuple

# настройки работы программы
do_original: bool = False
do_a_star: bool = True
do_brute_force: bool = False
do_nearest_neighbor: bool = True
do_genetic: bool = False

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
choose_points: int = 6  # 0 Генерировать ли случайные числа, 6 чисел, 10 чисел, 20 чисел

# Константа для коэффициента перепада высоты
height_cost_coefficient: float = 100

# Параметры генетического метода
population_size: int = 100  # Размер популяции
generations: int = 100  # Количество поколений
mutation_rate: float = 0.3  # Вероятность мутации
tournament_size: int = population_size // 10  # Количество агентов для отбора
