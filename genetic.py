import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 

def getRandomGraph(N):
    """
        Generate a random, symmetric and geographic graph
        Parameters: 
            N: Number of cities 
        Return:
            python dictionary: The key is the edge and the value is the weight 
    """
    x, y = np.random.rand(2, N)
    num_elements = int(N*(N-1)/2)
    W = {}
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            W.update({(i,j) : d})
            W.update({(j,i) : d})
    return W, x, y

def generateInitialConf(numConfs, N):
    """
        Generate random confs
        Parameters:
            N: number of configurations 
        Return:
            List of random configurations 
    """
    aux = [i for i in range(N)]
    listConfs = []
    while len(listConfs) < numConfs:
        aux = np.random.choice(aux, N, replace=False).tolist()
        if aux not in listConfs: listConfs.append(aux)
    return listConfs

def calculateObjetiveFunct(conf, N, W):
    """
        Returns the length of a trajectory 
        Parameters:
            conf: A route (configuration)
            N: number of cities 
            W: edge dictionary 
        Returns:
            float represent the route distance 
    """
    edges = [(conf[i], conf[(i+1)%N]) for i in range(N)]
    return np.sum([ W[edge] for edge in edges])

def mutation(conf, N):
    """
        This function makes a permutation in the route 
        Parameters:
            conf: a route 
            N: route length 
        Return:
            The mutated route 
    """
    aux = np.arange(N)
    i, j = np.random.choice(aux, 2, replace=False)
    aux = []
    for pos in range(N):
        if pos == i: aux.append(conf[j])
        elif pos == j: aux.append(conf[i])
        else: aux.append(conf[pos])
    return aux

def crossover(conf1, conf2, N):
    """
        Performs a crossover operation

        Parameters: 
            conf1, conf2: Valid routes 
            N: route length 
        return: 
            A route offspring 
    """
    child = []
    childP1 = []
    childP2 = []

    geneA = int(np.random.random() * N)
    geneB = int(np.random.random() * N)

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(conf1[i])

    childP2 = [item for item in conf2 if item not in childP1]

    child = childP1 + childP2
    return child 



def Darwing(N, steeps, numIndividuals, numMutations, numCrossovers):
    """
        Do the evolution 

        Parameters:
            N: number of cities 
            steps: Number of iterations 
            numIndividuals 
            numMutations: number of chromosomes that will mutate 
            numCrossovers: number of chromosomes that will generate an offspring 
        
        Returns: 
            The shorter route of each iteration

    """
    auxNumIndividuals = np.arange(numIndividuals)
    best = []

    W, x, y  = getRandomGraph(N)
    population = generateInitialConf(numIndividuals, N)  
    lengths = [(calculateObjetiveFunct(conf, N, W), conf) for conf in population]
    lengths.sort()

    for time in range(steeps):
        print(time)
        mutation_crossover = np.random.choice(auxNumIndividuals, 2*numCrossovers + numMutations, replace=False)

        mutated = []
        for i, m in enumerate(mutation_crossover[:numMutations]):
            mutated.append(mutation(population[m], N))
        
        offspring = []
        parents1 = mutation_crossover[numMutations:numMutations + numCrossovers]
        parents2 = mutation_crossover[numMutations + numCrossovers:]
        for i in range(numCrossovers):
            offspring.append(crossover(population[parents1[i]], population[parents2[i]], N))
        
        allIndividuals = population +  mutated + offspring
        lengths = [(round(calculateObjetiveFunct(conf, N, W), 6), conf) for conf in allIndividuals]
        
        auxDict = {i[0]:i[1] for i in lengths}
        lengths = [(i, auxDict[i]) for i in auxDict.keys()]
        lengths.sort()

        population = [ i[1] for i in lengths]
        population = population[:numIndividuals]

        best.append((lengths[1]))


    return best, W, x, y


def plotRoute(conf, W, x, y, N):
    """
        Plot a networkx graph containing a route 
        Parameters:
            conf: A list representing a configuration
            W: a dictionary representing the graph
            x, y: node positions 
            N: number of nodes 
    """
    route = [(conf[i], conf[(i+1)%N]) for i in range(N)]
    pos = {i:(x[i], y[i]) for i in range(N)}
    G = nx.Graph()
    G.add_edges_from(W)
    
    nx.draw_networkx_nodes(G, pos)  
    nx.draw_networkx_edges(G, pos, route)






