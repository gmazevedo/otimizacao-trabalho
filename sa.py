import random
import math
from collections import defaultdict
import sys
from time import time
import copy

def main():
    """ Recebe parâmetros de entrada """
    OUTPUT_FILE = sys.argv[1]
    INSTANCE_FILE = sys.argv[2]
    COOLING_FACTOR = sys.argv[3]
    ITERATIONS = sys.argv[4]
    INITIAL_PROB = sys.argv[5]
    FINAL_PROB = sys.argv[6]
    SEED = sys.argv[7]

    seed = SEED
    out_file = OUTPUT_FILE
    instance = INSTANCE_FILE
    r = float(COOLING_FACTOR)
    itera = int(ITERATIONS)
    pi = float(INITIAL_PROB)
    pf = float(FINAL_PROB)
    
    # Seed para gerar números aleatórios
    random.seed(seed)

    # Gera uma instância de um grafo a partir do arquivo de entrada INSTANCE_FILE
    num_vertex, num_edges, num_colors, weights, edges = set_grafo(instance)

    # Executa Simulated Annealing dentro de cmb()
    solution, initial_value, best_value, initial_num_conflicts, final_num_conflicts, time_elapsed = cmb(num_vertex, num_edges, num_colors, weights, edges, r, itera, pi, pf)

    # Grava o resultado em OUTPUT_FILE
    write_output(out_file,instance,num_vertex,num_edges,r,itera,pi,pf,time_elapsed,seed,initial_value,best_value,initial_num_conflicts,final_num_conflicts,solution)


def set_grafo(filename):
    file = open(filename, 'r')

    weights = []

    current_line = 1

    for line in file:
        instance = line.strip().split()

        if current_line == 1:
            num_vertex = int(instance[0])
            num_edges = int(instance[1])
            num_colors = int(instance[2])
            edges = [[0 for x in range(num_vertex)] for y in range(num_vertex)]
        elif current_line == 2:
            for i in range(0, num_vertex):
                weights.append(float(instance[i]))
        elif (current_line > 2) and (current_line < num_edges+2):
            u = int(instance[0])
            v = int(instance[1])
            edges[u][v] = 1

        current_line = current_line + 1

    file.close()

    return num_vertex, num_edges, num_colors, weights, edges

def cores_vertices(vertex, solution, num_vertex, num_colors):
    found = 0
    color = 0

    while found == 0:
        if solution[vertex][color] == 1:
            found = 1
        else:
            color = color + 1

    return color

def cont_conflitos(solution, num_vertex, num_edges, num_colors, edges):
    color_conflicts = 0

    for u in range(0,num_vertex):
        for v in range(0,num_vertex):
            if edges[u][v] == 1:
                u_color = cores_vertices(u,solution,num_vertex,num_edges)
                v_color = cores_vertices(v,solution,num_vertex,num_edges)
                if u_color == v_color:
                    color_conflicts = color_conflicts + 1

    return color_conflicts

def random_zero_um():
    return random.uniform(0.0,1.0)


def sol_inicial(num_vertex, num_edges, num_colors):
    solution = [[0 for x in range(num_vertex)] for y in range(num_vertex)]

    for vertex in range(0,num_vertex):
        color = random.randint(0,num_colors-1)
        solution[vertex][color] = 1

    return solution

def peso_sol(solution, num_vertex, num_edges, num_colors, weights, edges):
    values = []

    for i in range(0,num_colors):
        values.append(0.0)

    color_conflicts = cont_conflitos(solution,num_vertex,num_edges,num_colors,edges)

    for color in range(0,num_colors):
        for vertex in range(0,num_vertex):
            if solution[vertex][color] == 1:
                values[color] = values[color] + weights[vertex]

    final_value = max(values) + (color_conflicts * 10000)
    return final_value, values

def gera_vizinho(solution, num_vertex, num_edges, num_colors):
    novaSol = copy.deepcopy(solution)

    vertex = random.randint(0,num_vertex-1)
    color = cores_vertices(vertex,novaSol,num_vertex,num_edges)

    new_color = random.randint(0,num_colors-1)

    while new_color == color:
        new_color = random.randint(0,num_colors-1)

    novaSol[vertex][color] = 0
    novaSol[vertex][new_color] = 1

    return novaSol

def temp_inicial(solution, temperatura, r, itera, initial_prob, num_vertex, num_edges, num_colors, weights, edges):
    pesoAtual, pesol_sol = peso_sol(solution,num_vertex, num_edges, num_colors, weights, edges)
    aprox_temp = temperatura
    aprox_prob = 1

    while(temperatura > 0.0001):
        movFact = 0
        tentativaMov = 0

        for i in range(0,itera):
            novaSol = gera_vizinho(solution,num_vertex,num_edges,num_colors)
            pesoCandidato, pesol_sol = peso_sol(novaSol,num_vertex, num_edges, num_colors, weights, edges)
            
            delta =  pesoCandidato - pesoAtual

            if delta <= 0:
                solution = novaSol
                pesoAtual = pesoCandidato
            else:
                tentativaMov = tentativaMov + 1
                prob = math.exp((-delta)/temperatura) # e^(delta/t)
                rand = random_zero_um() # número entre 0 e 1
                if rand < prob:
                    movFact = movFact + 1
                    solution = novaSol
                    pesoAtual = pesoCandidato
        if tentativaMov > 0: # se for movimento de subida
            if ((initial_prob - 0.005) <= (movFact/tentativaMov)) and ((movFact/tentativaMov) <= (initial_prob + 0.005)):
                return temperatura
            elif abs((movFact/tentativaMov) - initial_prob) < abs(aprox_prob - initial_prob):
                    aprox_temp = temperatura
                    aprox_prob = (movFact/tentativaMov)

        temperatura = temperatura * r

    return aprox_temp


def simulated_annealing(solution, temperatura, r, itera, pf, num_vertex, num_edges, num_colors, weights,edges):
    pesoAtual, pesol_sol = peso_sol(solution,num_vertex, num_edges, num_colors, weights,edges)
    contPf = 0

    while(contPf < 5):
        movFact = 0
        tentativaMov = 0

        for i in range(0,itera):
            novaSol = gera_vizinho(solution,num_vertex,num_edges,num_colors)
            pesoCandidato, pesol_sol = peso_sol(novaSol,num_vertex, num_edges, num_colors, weights, edges)

            delta =  pesoCandidato - pesoAtual

            if delta <= 0: # se for movimento de descida
                solution = novaSol
                pesoAtual = pesoCandidato
                contPf = 0
            else:
                tentativaMov = tentativaMov + 1
                prob = math.exp((-delta)/temperatura) # e^(delta/t)
                rand = random_zero_um()
                if rand < prob:
                    movFact = movFact + 1
                    solution = novaSol
                    pesoAtual = pesoCandidato

        if tentativaMov > 0 and pf > (movFact/tentativaMov):
            contPf = contPf + 1 # esfria

        temperatura = temperatura * r # reduz a temperatura

    return solution

def cmb(num_vertex, num_edges, num_colors, weights, edges, r, itera, initial_prob, pf):
    """ Implementação do algoritmo CMB """
    init_temp_temperature = 100
    init_temp_iterations = 10

    best_value = 0

    start = time()

    print("\nSolução inicial")
    initial_solution = sol_inicial(num_vertex, num_edges, num_colors)
    initial_value, pesol_sol = peso_sol(initial_solution, num_vertex, num_edges, num_colors, weights,edges)
    initial_num_conflicts = cont_conflitos(initial_solution,num_vertex,num_edges,num_colors,edges)

    print("Valor = " + str(initial_value))
    print("Conflito de cores  = {}".format(initial_num_conflicts))

    print("\nTemperatura inicial")
    initial_teperature = temp_inicial(initial_solution, init_temp_temperature, r, init_temp_iterations, initial_prob, num_vertex, num_edges, num_colors, weights, edges)
    print(str(initial_teperature))

    print("\nExecutando solução final")
    solution = simulated_annealing(initial_solution, initial_teperature, r, itera, pf, num_vertex, num_edges, num_colors, weights, edges)

    best_value, pesol_sol = peso_sol(solution, num_vertex, num_edges, num_colors, weights,edges)
    print("\nValor da solução final = " + str(best_value))

    final_num_conflicts = cont_conflitos(solution,num_vertex,num_edges,num_colors,edges)
    print("Numero de conflito de cores = {}".format(final_num_conflicts))
    end = time()
    time_elapsed = end - start

    return solution, initial_value, best_value, initial_num_conflicts, final_num_conflicts, time_elapsed


def write_output(out_filename, instance_filename, num_vertices, num_edges, r, I, initial_prob, final_prob, time_elapsed, seed, initial_value, best_value, initial_num_conflicts, final_num_conflicts, solution):
    """ Grava resultado - CMB utilizando Simulated Annealing  """
    out_file = open(out_filename, "w")

    out_file.write("Instance File:\n  " + instance_filename + "\n")
    out_file.write("\nTime:\n  " + str(time_elapsed) + "\n")
    out_file.write("\nParameters:\n")
    out_file.write("  Cooling Factor r: " + str(r) + "\n")
    out_file.write("  Instances I: " + str(I) + "\n")
    out_file.write("  Initial probability: " + str(initial_prob) + "\n")
    out_file.write("  Final probability: " + str(final_prob) + "\n")
    out_file.write("  Seed: " + str(seed) + "\n")
    out_file.write("\nInitial Solution Value:\n  " + str(initial_value) + "\n")
    out_file.write("\nInitial number of color conflicts:\n  " + str(initial_num_conflicts) + "\n")
    out_file.write("\nFinal Solution Value:\n  " + str(best_value) + "\n")
    out_file.write("\nFinal number of color conflicts:\n  " + str(final_num_conflicts) + "\n")
    out_file.write("\nSolution:\n")
    for vertex in range(0,num_vertices):
        color = cores_vertices(vertex,solution,num_vertices,num_edges)
        if solution[vertex][color] == 1:
            out_file.write("  Vertex {} color = {}\n".format(vertex,color))
    out_file.close()

##### APRESENTAÇÃO DE RESULTADOS ####
def print_instance(num_vertex, num_edges, num_colors, weights, edges):
    print("------- Intância ------- ")
    print(" ")
    print(" Vertices: {} ".format(num_vertex))
    print(" Arestas: {} ".format(num_edges))
    print(" Cores: {} ".format(num_colors))
    print(" ")

    for i in range(0, num_vertex):
        print(" vertex {} weight: {}".format(i,weights[i]))

    print(" ")
    print(" {} edges:".format(num_edges))
    for u in range(0,num_vertex):
        for v in range(0,num_vertex):
            if edges[u][v] == 1:
                print(" [{}][{}]".format(u,v))
    print(" ")

def print_solution(solution, num_vertex, num_colors):
    print("------- Solução ------- ")
    print(" ")

    for vertex in range(0,num_vertex):
        color = cores_vertices(vertex,solution,num_vertex,num_colors)
        print(" Vertice {} Cor = {}".format(vertex,color))

    print(" ")

def print_color_values(pesol_sol, num_colors):
    print("----- Cores e valores ----- ")
    print(" ")

    print(" Valor máximo da cor:")
    print(" " + str(max(pesol_sol)) + "\n ")

    for color in range(0,num_colors):
        print(" Valor da cor {} = {}".format(color,pesol_sol[color]))
    print(" ")
    
############ MAIN #############
main()



