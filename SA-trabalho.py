#!/usr/bin/env python
# coding: utf-8

# # Bibliotecas usadas

# In[1]:


import random
import math
from collections import defaultdict
import sys
from time import time
import copy


# In[2]:


OUTPUT_FILE = sys.argv[1]
INSTANCE_FILE = sys.argv[2]
COOLING_FACTOR = sys.argv[3]
ITERATIONS = sys.argv[4]
INITIAL_PROB = sys.argv[5]
FINAL_PROB = sys.argv[6]
SEED = sys.argv[7]


# # Formato do arquivo que contém o grafo
# * a line with n, m, k
# * a line with n real numbers, corresponding to the vertex weights
# * m lines, each with 2 vertex indices (base 0), representing the edges
# * EXEMPLE 
# 
#  3 3 3
#  
#  10.1 21.2 38.3
#  
#  10.1 21.2
#  
#  10.1 38.3
#  
#  38.3 21.2

# # Funções usadas

# In[3]:


def instancia(filename):
    file = open(filename, 'r')

    pesos = []

    current_line = 1

    for line in file:
        instance = line.strip().split()

        if current_line == 1:
            numVertices = int(instance[0])
            numArestas = int(instance[1])
            num_colors = int(instance[2])
            edges = [[0 for x in range(numVertices)] for y in range(numVertices)]

        if current_line == 2:
            for i in range(0, numVertices):
                pesos.append(float(instance[i]))

        if (current_line > 2) and (current_line <= (numArestas+2)):
            u = int(instance[0])
            v = int(instance[1])
            edges[u][v] = 1

        current_line = current_line + 1

    file.close()

    return numVertices, numArestas, num_colors, pesos, edges


# In[4]:


def cores_vertices(vertex, solution, numVertices, num_colors):
    found = 0
    color = 0

    while found == 0:
        if solution[vertex][color] == 1:
            found = 1
        else:
            color = color + 1

    return color


# In[5]:


def cont_conflitos(solution, numVertices, numArestas, num_colors, edges):
    color_conflicts = 0

    for u in range(0,numVertices):
        for v in range(0,numVertices):
            if edges[u][v] == 1:
                u_color = cores_vertices(u,solution,numVertices,numArestas)
                v_color = cores_vertices(v,solution,numVertices,numArestas)
                if u_color == v_color:
                    color_conflicts = color_conflicts + 1

    return color_conflicts


# In[6]:


def random_zero_um():
    return random.randrange(0,1)


# In[7]:


def sol_inicial(numVertices, numArestas, num_colors):
    solution = [[0 for x in range(numVertices)] for y in range(numVertices)]

    for vertex in range(0,numVertices):
        color = random.randint(0,num_colors-1)
        solution[vertex][color] = 1

    return solution


# In[8]:


def peso_sol(solution, numVertices, numArestas, num_colors, pesos):
    values = []

    for i in range(0,num_colors):
        values.append(0.0)

    color_conflicts = cont_conflitos(solution,numVertices,numArestas,num_colors,edges)

    for color in range(0,num_colors):
        for vertex in range(0,numVertices):
            if solution[vertex][color] == 1:
                values[color] = values[color] + pesos[vertex]

    final_value = max(values) + (color_conflicts * 10000)
    return final_value, values


# In[9]:


def gera_vizinho(solution, numVertices, num_colors):
    novaSol = copy.deepcopy(solution)

    vertex = random.randint(0,numVertices-1)
    color = cores_vertices(vertex,novaSol,numVertices,numArestas)

    new_color = random.randint(0,num_colors-1)

    while new_color == color:
        new_color = random.randint(0,num_colors-1)

    novaSol[vertex][color] = 0
    novaSol[vertex][new_color] = 1

    return novaSol


# In[10]:


def temp_inicial(solution, temperatura, r, itera, initial_prob, numVertices, numArestas, num_colors, pesos):
    pesoAtual, pesol_sol = peso_sol(solution,numVertices, numArestas, num_colors, pesos)
    aprox_temp = temperatura
    aprox_prob = 1

    while(temperatura > 0.0001):
        movFact = 0
        tentativaMov = 0
        for i in range(0,itera):
            novaSol = gera_vizinho(solution,numVertices,num_colors)
            pesoCandidato, pesol_sol = peso_sol(novaSol,numVertices, numArestas, num_colors, pesos)

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
            else:
                if abs((movFact/tentativaMov) - initial_prob) < abs(aprox_prob - initial_prob):
                    aprox_temp = temperatura
                    aprox_prob = (movFact/tentativaMov)
        temperatura = temperatura * r

    return aprox_temp


# In[11]:


def set_grafo(filePath):
    """ Implementação da leitura de um grafo. """
    grafo = Grafo()
    
    with open("filepath",r) as newFile:
        grafoRaw = newFile.readlines()

        for line in grafoRaw:   
            if len (line.split()) == 3:
                grafo.adiciona_nmk(line.split())
            elif len (line.split()) > 3:
                grafo.adiciona_peso(line.split())
            elif len (line.split()) == 2:
                grafo.adiciona_arestas(line.split())

    return grafo


# In[12]:


class Grafo(object):
    """ Implementação básica de um grafo. """

    def __init__(self, arestas, direcionado=False):
        """Inicializa as estruturas base do grafo."""
        self.adj = defaultdict(set)
        self.direcionado = direcionado
        self.adiciona_arestas(arestas)
        self.nmk = []
        self.pesos = defaultdict(set)

    def get_vertices(self):
        """ Retorna a lista de vértices do grafo. """
        return list(self.adj.keys())

    def get_arestas(self):
        """ Retorna a lista de arestas do grafo. """
        return [(k, v) for k in self.adj.keys() for v in self.adj[k]]
    
    def get_pesos(self):
        """ Retorna a lista de pesos dos vértices do grafo. """
        return self.pesos
   
    def get_nmk(self):
        """ Retorna a lista com os valores de n, m e k do grafo. """
        return self.nmk

    def adiciona_arestas(self, arestas):
        """ Adiciona arestas ao grafo. """
        for u, v in arestas:
            self.adiciona_arco(u, v)

    def adiciona_arco(self, u, v):
        """ Adiciona uma ligação (arco) entre os nodos 'u' e 'v'. """
        self.adj[u].add(v)
        # Se o grafo é não-direcionado, precisamos adicionar arcos nos dois sentidos.
        if not self.direcionado:
            self.adj[v].add(u)
    
    def adiciona_nmk(self, nmk):
        """ Adiciona valores de n, m e k ao grafo. """
        self.nmk = nmk
        
    def adiciona_pesos(self, pesos):
        """ Cria lista de pesos dos vértices """
        i = 0
        for peso in pesos:
            self.pesos[i].add(peso) 

    def existe_aresta(self, u, v):
        """ Existe uma aresta entre os vértices 'u' e 'v'? """
        return u in self.adj and v in self.adj[u]


    def __len__(self):
        return len(self.adj)


    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self.adj))


    def __getitem__(self, v):
        return self.adj[v]


# # Implementação do método Simulated Annealing

# In[13]:


def simulated_annealing(solution, temperatura, r, itera, pf, numVertices, numArestas, num_colors, pesos):
    pesoAtual, pesol_sol = peso_sol(solution,numVertices, numArestas, num_colors, pesos)
    contPf = 0

    while(contPf < 5):
        movFact = 0
        tentativaMov = 0
        for i in range(0,itera):
            novaSol = gera_vizinho(solution,numVertices,num_colors)
            pesoCandidato, pesol_sol = peso_sol(novaSol,numVertices, numArestas, num_colors, pesos)

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
        if tentativaMov > 0:
            if pf > (movFact/tentativaMov):
                contPf = contPf + 1 # esfria
        temperatura = temperatura * r # reduz a temperatura

    return solution


# In[14]:


def cmb(numVertices, numArestas, num_colors, pesos, edges, r, itera, initial_prob, pf):
    init_temp_temperature = 10000
    init_temp_iterations = 100

    best_value = 0

    start = time()

    print("solução inicial")
    initial_solution = sol_inicial(numVertices, numArestas, num_colors)
    initial_value, pesol_sol = peso_sol(initial_solution, numVertices, numArestas, num_colors, pesos)
    initial_num_conflicts = cont_conflitos(initial_solution,numVertices,numArestas,num_colors,edges)

    print(" Valor = " + str(initial_value))
    print(" conflito de cores  = {}".format(initial_num_conflicts))

    print("\n temperatura inicial")
    initial_teperature = temp_inicial(initial_solution, init_temp_temperature, r, init_temp_iterations, initial_prob, numVertices, numArestas, num_colors, pesos)


    print("solução final")
    solution = simulated_annealing(initial_solution, initial_teperature, r, itera, pf, numVertices, numArestas, num_colors, pesos)


    best_value, pesol_sol = peso_sol(solution, numVertices, numArestas, num_colors, pesos)
    print(" valor da solução final = " + str(best_value) + "\n")

    final_num_conflicts = cont_conflitos(solution,numVertices,numArestas,num_colors,edges)
    print("numero de conflito de cores = {}".format(final_num_conflicts))
    print("\n")
    end = time()
    time_elapsed = end - start

    return solution, initial_value, best_value, initial_num_conflicts, final_num_conflicts, time_elapsed


# In[15]:


def write_output(out_filename,instance_filename, num_vertices, num_edges, r, I, initial_prob, final_prob, time, seed, initial_value, best_value, initial_num_conflicts,final_num_conflicts, solution):
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


# In[16]:


def print_instance(numVertices, numArestas, num_colors, pesos, edges):
    print("------- Intância ------- ")
    print(" ")
    print(" Vertices: {} ".format(numVertices))
    print(" Arestas: {} ".format(numArestas))
    print(" Cores: {} ".format(num_colors))
    print(" ")

    for i in range(0, numVertices):
        print(" vertex {} weight: {}".format(i,pesos[i]))

    print(" ")
    print(" {} edges:".format(numArestas))
    for u in range(0,numVertices):
        for v in range(0,numVertices):
            if edges[u][v] == 1:
                print(" [{}][{}]".format(u,v))
    print(" ")


# In[17]:


def print_solution(solution, numVertices, num_colors):
    print("------- Solução ------- ")
    print(" ")

    for vertex in range(0,numVertices):
        color = cores_vertices(vertex,solution,numVertices,num_colors)
        print(" Vertice {} Cor = {}".format(vertex,color))

    print(" ")


# In[18]:


def print_color_values(pesol_sol, num_colors):
    print("----- Cores e valores ----- ")
    print(" ")

    print(" Valor máximo da cor:")
    print(" " + str(max(pesol_sol)) + "\n ")

    for color in range(0,num_colors):
        print(" Valor da cor {} = {}".format(color,pesol_sol[color]))
    print(" ")


# In[19]:


seed = SEED
out_file = OUTPUT_FILE
instance = INSTANCE_FILE
r = float(COOLING_FACTOR)
itera = int(ITERATIONS)
pi = float(INITIAL_PROB)
pf = float(FINAL_PROB)

random.seed(seed)
numVertices, numArestas, num_colors, pesos, edges = instancia(instance)
solution, initial_value, best_value, initial_num_conflicts, final_num_conflicts, time_elapsed = cmb(numVertices, numArestas, num_colors, pesos, edges, r, itera, pi, pf)
write_output(out_file,instance,numVertices,numArestas,r,itera,pi,pf,time_elapsed,seed,initial_value,best_value,initial_num_conflicts,final_num_conflicts,solution)


# In[ ]:




