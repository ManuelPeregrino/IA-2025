import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Definir la función f(x)
def f(x):
    return 0.1 * x * np.log(1 + np.abs(x)) * np.cos(x) * np.cos(x)

# Parámetros del intervalo
A = 5  # Límite inferior
B = 100  # Límite superior
delta_x = 0.01  # Resolución original

# Parámetros del algoritmo genético
max_poblacion = 30
min_poblacion = 20
p_cruza = 0.8
p_mut_individuo = 0.7
p_mut_bit = 0.6
elitismo = True

# Calcular el número de puntos
n = int((B - A) / delta_x) + 1

# Calcular el número de bits necesarios
num_bits = math.ceil(math.log2(n))

# Calcular la resolución interna del sistema
Delta_x_star = (B - A) / (2**num_bits - 1)

# Generar los índices y transformarlos a valores de x usando Delta_x_star
indices = np.arange(2**num_bits)
puntos = A + indices * Delta_x_star

# Filtrar los puntos dentro del rango útil (excluir los que exceden n)
puntos_utiles = puntos[:n]

# Calcular los valores de la función f(x) en cada punto útil
valores_f = f(puntos_utiles)

# Codificar los puntos útiles en binario
codigos_binarios = [format(i, f'0{num_bits}b') for i in range(n)]

# Inicializar población aleatoria
poblacion_inicial = random.sample(codigos_binarios, max_poblacion)

# Función para formar parejas mediante selección por torneo
def formar_parejas(valores_f, p_cruza):
    parejas = []
    for _ in range(len(valores_f) // 2):
        torneo = random.sample(range(len(valores_f)), 4)
        p1 = max(torneo[:2], key=lambda i: valores_f[i])
        p2 = max(torneo[2:], key=lambda i: valores_f[i])
        if random.uniform(0, 1) <= p_cruza:
            parejas.append((p1, p2))
    return parejas

# Función para realizar el cruzamiento
def cruzar_parejas(parejas, codigos_binarios):
    descendientes = []
    for i, j in parejas:
        padre = codigos_binarios[i]
        madre = codigos_binarios[j]
        punto_cruce = random.randint(1, len(padre) - 1)
        hijo1 = padre[:punto_cruce] + madre[punto_cruce:]
        hijo2 = madre[:punto_cruce] + padre[punto_cruce:]
        descendientes.append(hijo1)
        descendientes.append(hijo2)
    return descendientes

# Función para realizar la mutación
def mutar_descendientes(descendientes, p_mut_individuo, p_mut_bit):
    descendientes_mutados = []
    for descendiente in descendientes:
        if random.uniform(0, 1) <= p_mut_individuo:
            descendiente_mutado = ''.join('1' if bit == '0' else '0' if random.uniform(0, 1) <= p_mut_bit else bit for bit in descendiente)
            descendientes_mutados.append(descendiente_mutado)
        else:
            descendientes_mutados.append(descendiente)
    return descendientes_mutados

# Función para realizar la poda con elitismo opcional
def poda(poblacion_actual, descendientes, valores_f, max_poblacion, elitismo):
    poblacion_combinada = poblacion_actual + descendientes
    poblacion_unica = list(set(poblacion_combinada))
    valores_x = [A + int(individuo, 2) * Delta_x_star for individuo in poblacion_unica]
    aptitudes = [f(x) for x in valores_x]

    individuos_ordenados = [x for _, x in sorted(zip(aptitudes, poblacion_unica), reverse=True)]
    if elitismo:
        mejor_individuo = poblacion_actual[np.argmax(valores_f)]
        if mejor_individuo not in individuos_ordenados:
            individuos_ordenados.insert(0, mejor_individuo)
    return individuos_ordenados[:max_poblacion]

# Definir el número de generaciones
num_generaciones = 50

# Inicializar población
poblacion_actual = poblacion_inicial

# Historial de evolución
historial_mejor = []
historial_promedio = []

for generacion in range(num_generaciones):
    valores_x = [A + int(individuo, 2) * Delta_x_star for individuo in poblacion_actual]
    aptitudes = [f(x) for x in valores_x]

    mejor_aptitud = max(aptitudes)
    promedio_aptitud = np.mean(aptitudes)
    mejor_x = valores_x[np.argmax(aptitudes)]

    historial_mejor.append((mejor_x, mejor_aptitud))
    historial_promedio.append(promedio_aptitud)

    parejas = formar_parejas(aptitudes, p_cruza)
    descendientes = cruzar_parejas(parejas, poblacion_actual)
    descendientes_mutados = mutar_descendientes(descendientes, p_mut_individuo, p_mut_bit)

    poblacion_actual = poda(poblacion_actual, descendientes_mutados, aptitudes, max_poblacion, elitismo)

# Resultados finales
mejor_x_final, mejor_aptitud_final = max(historial_mejor, key=lambda x: x[1])
print(f"Mejor x encontrado: {mejor_x_final:.4f}")
print(f"Mejor f(x) encontrado: {mejor_aptitud_final:.4f}")

# Graficar resultados
generaciones = list(range(1, num_generaciones + 1))
mejores_f = [valor for _, valor in historial_mejor]
plt.figure(figsize=(10, 6))
plt.plot(generaciones, mejores_f, label="Mejor f(x) por generación", marker='o', color='blue', linewidth=2)
plt.plot(generaciones, historial_promedio, label="Aptitud promedio por generación", linestyle='--', color='green', linewidth=2)
plt.title("Evolución del Algoritmo Genético")
plt.xlabel("Generación")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()