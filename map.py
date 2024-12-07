import noise
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import List, Tuple
import random


# Генерация карты
def generate_map(_width: int, _height: int, _scale: float, _octaves: int, _persistence: float,
                 _lacunarity: float) -> np.ndarray:
    map_data: np.ndarray = np.zeros((_height, _width))
    for y in range(_height):
        for x in range(_width):
            # Генерация значения шума Перлина в диапазоне [-1, 1]
            raw_value = noise.snoise2(
                x / _scale,
                y / _scale,
                octaves=_octaves,
                persistence=_persistence,
                lacunarity=_lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=42
            )

            # Нормализация в диапазон [0, 1]
            normalized_value = (raw_value + 1) / 2  # Переводим в диапазон [0, 1]

            # Записываем в карту
            map_data[y][x] = normalized_value

    return map_data


def prepare_basic_map(_start: Tuple[int, int], _end: Tuple[int, int], _points: List[Tuple[int, int]],
                      _terrain_map: np.ndarray) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(_terrain_map, cmap='terrain')
    plt.colorbar(label="Высота")

    # Точки маршрута (пурпурные, большие, с кругами)
    plt.scatter([p[1] for p in _points], [p[0] for p in _points], color="purple", s=100,
                marker='o', label="Точки маршрута", zorder=3, alpha=0.7, edgecolor='black')

    # Начальная точка (зеленая, большая, с треугольником)
    plt.scatter(_start[1], _start[0], color="green", s=150, marker='^', label="Начальная точка", zorder=3,
                edgecolor='black', linewidth=2)

    # Конечная точка (красная, большая, с квадратом)
    plt.scatter(_end[1], _end[0], color="red", s=150, marker='s', label="Конечная точка", zorder=3,
                edgecolor='black', linewidth=2)


def draw_path(_path: List[Tuple[int, int]]) -> None:
    for i in range(len(_path) - 1):
        # Рисуем линии между точками
        plt.plot([_path[i][1], _path[i + 1][1]], [_path[i][0], _path[i + 1][0]], color="red", linewidth=1, zorder=1)

        if i % 30 == 0:  # Рисуем стрелки каждые 30 шагов

            # Используем FancyArrowPatch для более плавных стрелок
            arrow = FancyArrowPatch(
                (_path[i][1], _path[i][0]),  # Начало стрелки
                (_path[i + 1][1], _path[i + 1][0]),  # Направление стрелки
                mutation_scale=10,  # Масштаб стрелки
                color="blue",  # Цвет стрелки
                arrowstyle="->",  # Стиль стрелки
                lw=2,  # Толщина стрелки
                alpha=0.7,  # Прозрачность
                zorder=2  # Порядок наложения
            )
            plt.gca().add_patch(arrow)


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
