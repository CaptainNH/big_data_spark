import matplotlib.pyplot as plt
import numpy as np

# Данные из ваших экспериментов
experiments = [
    "1 DN, без оптимизации",
    "1 DN, с оптимизацией",
    "3 DN, без оптимизации",
    "3 DN, с оптимизацией"
]

time_results = [61.73, 60.46, 64.83, 69.04]  # Время в секундах
memory_results = [36.00, 36.00, 36.00, 36.00]  # Память в MB

# Создаем фигуру с двумя субплогами
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# График времени выполнения
bars1 = ax1.bar(experiments, time_results, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax1.set_title('Время выполнения (секунды)')
ax1.set_ylabel('Время (сек)')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.bar_label(bars1, fmt='%.2f', padding=3)

# График использования памяти
bars2 = ax2.bar(experiments, memory_results, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax2.set_title('Использование памяти (MB)')
ax2.set_ylabel('Память (MB)')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.bar_label(bars2, fmt='%.2f', padding=3)

# Общие настройки
plt.suptitle('Сравнение производительности разных конфигураций', fontsize=14)
plt.tight_layout()

# Добавим таблицу с данными под графиками
plt.figure(figsize=(10, 4))
columns = ('Конфигурация', 'Время (сек)', 'Память (MB)')
cell_text = [[exp, f"{time:.2f}", f"{mem:.2f}"]
             for exp, time, mem in zip(experiments, time_results, memory_results)]

table = plt.table(cellText=cell_text,
                 colLabels=columns,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.axis('off')

plt.show()