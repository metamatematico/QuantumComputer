#!/usr/bin/env python
# coding: utf-8

# """
# Este script está diseñado para resolver un problema de optimización cuántica en un tablero de 5x5 utilizando un solver de D-Wave.
# El proceso implica la iteración adiabática a través de diferentes valores del parámetro gamma, que controla la transición del 
# sistema cuántico desde un Hamiltoniano inicial hasta el Hamiltoniano final del problema. A continuación se explica cada sección del script:
# 
# 1. **Importación de Librerías**:
#    Se importan las librerías necesarias para el manejo de matrices, gráficos, y herramientas específicas de D-Wave y PyQUBO.
# 
# 2. **Configuración del Solver**:
#    Se configura el solver cuántico de D-Wave con el tipo de solver especificado y el token de autenticación.
# 
# 3. **Parámetros Iniciales del Modelo**:
#    Se definen los parámetros iniciales como el tamaño del tablero, la fuerza de la cadena para el embedding, y las energías asociadas a las piedras negras y blancas.
# 
# 4. **Creación de Variables de Spin**:
#    Se crean variables de Spin para cada posición en el tablero utilizando la biblioteca PyQUBO.
# 
# 5. **Definición del Hamiltoniano Cuántico**:
#    Se define una función que construye el Hamiltoniano cuántico para una posición dada en el tablero considerando sus vecinos, y se añaden términos transversales proporcionales a gamma.
# 
# 6. **Construcción y Resolución del Hamiltoniano**:
#    Se construye el Hamiltoniano total iterando sobre todas las posiciones del tablero y se resuelve usando el solver cuántico.
# 
# 7. **Evaluación de la Optimalidad**:
#    Se evalúa la energía total del sistema y se ajustan los coeficientes del Hamiltoniano (J_black y J_white) basándose en los resultados obtenidos.
# 
# 8. **Iteración Adiabática**:
#    Se realiza una iteración adiabática variando gamma desde 0 hasta el valor máximo especificado en pasos definidos, ajustando los coeficientes y almacenando las respuestas intermedias en cada paso.
# 
# 9. **Visualización y Análisis de Resultados**:
#    Se generan gráficos para visualizar la energía total frente a gamma y los ajustes de los coeficientes J_black y J_white. Además, se representa la evolución del proceso de optimización en un grafo.
# 
# El objetivo de este script es proporcionar una solución óptima al problema de optimización en el tablero, aprovechando las capacidades del solver cuántico de D-Wave y ajustando dinámicamente los parámetros del Hamiltoniano para mejorar la solución a lo largo de las iteraciones adiabáticas.
# """
# 

# In[ ]:


get_ipython().system('pip install networkx')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
import dwave.inspector as ins
from pyqubo import Spin, Binary

# Configuración del solver
solver_type = 'Advantage2_prototype2.3'
dwave_token = "DEV-951435db241ce06d7f8f8ce960e5c44c5475f72d"
solver = DWaveSampler(solver=solver_type, token=dwave_token)

# Parámetros iniciales del modelo
params = {
    'size': 5,  # Tamaño del tablero.
    'chain_strength': 2.0,  # Fuerza de la cadena para embedding
    'J_black': -1.0,  # Energía para piedras negras
    'J_white': 1.0,  # Energía para piedras blancas
    'J_optimal': 0.0,  # Punto de optimalidad
    'gamma': 0.7,  # Coeficiente transversal
    'steps': 10  # Número de pasos adiabáticos
}

# Crear variables de Spin para cada posición en el tablero
spins = {(i, j): Spin(f's_{i}_{j}') for i in range(params['size']) for j in range(params['size'])}

# Hamiltoniano cuántico adaptado con condiciones de frontera
def create_quantum_hamiltonian(i, j):
    neighbors = [
        (i-1, j) if i > 0 else (i, j),
        (i+1, j) if i < params['size']-1 else (i, j),
        (i, j-1) if j > 0 else (i, j),
        (i, j+1) if j < params['size']-1 else (i, j)
    ]
    
    H_quantum = (
        params['J_optimal'] * spins[(i, j)] * spins[neighbors[0]] +  # PauliZ(0) @ PauliZ(1)
        params['J_black'] * spins[(i, j)] * spins[neighbors[1]] +  # PauliZ(0) @ PauliZ(1)
        params['J_white'] * spins[(i, j)] * spins[neighbors[2]] +  # PauliZ(0) @ PauliZ(1)
        params['J_optimal'] * spins[(i, j)] * spins[neighbors[3]]    # PauliZ(0) @ PauliZ(1) (simétrico)
    )
    
    return H_quantum

def add_transverse_terms(H, gamma):
    # Añadir términos transversales como términos de campo local
    for i in range(params['size']):
        for j in range(params['size']):
            H += gamma * spins[(i, j)]
    return H

def build_and_solve_hamiltonian(intermediate_gamma):
    global qubo, offset
    H = 0
    for i in range(params['size']):
        for j in range(params['size']):
            H += create_quantum_hamiltonian(i, j)

    # Añadir términos transversales al Hamiltoniano
    H = add_transverse_terms(H, intermediate_gamma)
    
    model = H.compile()
    qubo, offset = model.to_qubo()
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=offset)
    sampler = EmbeddingComposite(solver)
    response = sampler.sample(bqm, chain_strength=params['chain_strength'], num_reads=100)
    return response, model

def evaluate_optimality(response):
    first_sample = response.first.sample
    board_state = np.zeros((params['size'], params['size']))
    for (i, j), spin in spins.items():
        q_value = first_sample.get(f's_{i}_{j}', 0)
        s_value = 2 * q_value - 1  # Convertir de QUBO a Ising
        board_state[i, j] = s_value

    energy_black = 0
    energy_white = 0
    for i in range(params['size']):
        for j in range(params['size']):
            spin_value = board_state[i, j]
            if (i, j) in spins and spin_value == -1:  # Piedra negra
                energy_black += params['J_black']
            elif (i, j) in spins and spin_value == 1:  # Piedra blanca
                energy_white += params['J_white']

    total_energy = energy_black + energy_white
    return total_energy, energy_black, energy_white, board_state

def adjust_coefficients(response):
    total_energy, energy_black, energy_white, board_state = evaluate_optimality(response)
    print(f"Energía total: {total_energy}")
    print(f"Energía para piedras negras: {energy_black}")
    print(f"Energía para piedras blancas: {energy_white}")

    # Ajustar coeficientes en función de las energías calculadas
    if abs(energy_black) > abs(energy_white):
        if energy_black < 0:
            params['J_black'] += 0.1
        else:
            params['J_black'] -= 0.1
    else:
        if energy_white < 0:
            params['J_white'] += 0.1
        else:
            params['J_white'] -= 0.1

    return total_energy

# Iterar adiabáticamente a través de los pasos
gamma_values = np.linspace(0, params['gamma'], params['steps'])
intermediate_responses = []

J_black_values = []
J_white_values = []

# Crear un grafo para el árbol de optimización
G = nx.DiGraph()

for gamma in gamma_values:
    response, model = build_and_solve_hamiltonian(gamma)
    total_energy = adjust_coefficients(response)
    intermediate_responses.append((gamma, total_energy))
    J_black_values.append(params['J_black'])
    J_white_values.append(params['J_white'])

    # Agregar nodo al grafo
    node_id = len(G.nodes)
    G.add_node(node_id, gamma=gamma, energy=total_energy, board=response.first.sample)
    
    # Conectar con el nodo anterior
    if node_id > 0:
        G.add_edge(node_id - 1, node_id)

# Imprimir QUBO y la solución final
print("QUBO:", qubo)
df = pd.DataFrame(intermediate_responses, columns=['Gamma', 'Total Energy'])
print(df)

# Mostrar el inspector y proporcionar el enlace para el último paso
inspector_url = ins.show(response)
print(f"Inspector URL: {inspector_url}")

# Transformar resultados QUBO a representación Ising y mostrar el tablero final
first_sample = response.first.sample
board_state = np.zeros((params['size'], params['size']))
for (i, j), spin in spins.items():
    q_value = first_sample.get(f's_{i}_{j}', 0)
    s_value = 2 * q_value - 1  # Convertir de QUBO a Ising
    board_state[i, j] = s_value

print(board_state)

# Mostrar los coeficientes finales del Hamiltoniano
print("Coeficientes finales del Hamiltoniano:")
print(f"J_black: {params['J_black']}")
print(f"J_white: {params['J_white']}")

# Mostrar el Hamiltoniano completo
print("Hamiltoniano completo:")
print(model)

# Gráfico de energía total vs gamma
plt.figure(figsize=(10, 5))
plt.plot(df['Gamma'], df['Total Energy'], marker='o', linestyle='-')
plt.xlabel('Gamma')
plt.ylabel('Total Energy')
plt.title('Total Energy vs Gamma')
plt.grid(True)
plt.show()

# Gráfico de ajuste de coeficientes J_black y J_white
plt.figure(figsize=(10, 5))
plt.plot(gamma_values, J_black_values, marker='o', linestyle='-', label='J_black')
plt.plot(gamma_values, J_white_values, marker='o', linestyle='-', label='J_white')
plt.xlabel('Gamma')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Adjustment vs Gamma')
plt.legend()
plt.grid(True)
plt.show()

# Visualizar el árbol de optimización
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold')
labels = nx.get_node_attributes(G, 'energy')
nx.draw_networkx_labels(G, pos, labels, font_size=8)
plt.title('Optimization Tree')
plt.show()

