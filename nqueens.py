import random
from deap import tools
from deap import base
from deap import creator

# Problem parameter
n = 20 # how many queens on a board
generations = 100 # how many generations the evolution will take
pop_size = 300 # how many individuals in the population

# Define classes
creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize toolbox
toolbox = base.Toolbox()
# Since there is only one queen per line, individuals are represented as permutation
toolbox.register("permutation", random.sample, range(n), n)

# Structure initializer
# An individual is a list that represents the position of each queen
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.permutation)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function
def evalNQueens(individual):
    size = len(individual)
    # Evaluate by counting the number of conflicts on the diagonals
    left_diagonal = [0] * (2 * size - 1) # the number of diagonals in a n*n board is 2n-1
    right_diagonal = [0] * (2 * size - 1) # left diagonal = \, right diagonal = /
    # Sum the number of queens on each line
    for i in range(size):
        left_diagonal[i + individual[i]] += 1
        right_diagonal[size - 1 - i + individual[i]] += 1
    # Count the number of conflicts on each diagonal
    sum = 0
    for i in range(2 * size - 1):
        if left_diagonal[i] > 1:
            sum += left_diagonal[i] - 1
        if right_diagonal[i] > 1:
            sum += right_diagonal[i] - 1
    return sum,

# Crossover function
def cxPartiallyMatched(ind1, ind2):
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each index in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1 # make sure the two points aren't the same position
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1 # swap two points

    # Apply crossover between two cxpoints
    for i in range(cxpoint1, cxpoint2):
        # Store the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched values
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2

# Mutation function
def mutShuffleIndices(individual, indpb): # individual whose parameters are to be shuffled; independent mutation probability
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            swap_index = random.randint(0, size - 2)
            if swap_index >= i:
                swap_index += 1 # to make sure i != swap_index
            individual[i], individual[swap_index] = individual[swap_index], individual[i]
    return individual

# Registering functions
toolbox.register("evaluate", evalNQueens)
toolbox.register("mate", cxPartiallyMatched)
toolbox.register("mutate", mutShuffleIndices, indpb=2.0 / n)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    # Visualization
    gen = range(generations)
    avg_list = []
    min_list = []
    max_list = []

    # Initialize population
    pop = toolbox.population(n=pop_size)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Begin the evolution
    for g in gen:
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offsprings
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        g_min = min(fits)
        g_max = max(fits)

        avg_list.append(mean)
        min_list.append(g_min)
        max_list.append(g_max)

        print("  Min %s" % g_min)
        print("  Max %s" % g_max)
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, with fitness values of %s" % (best_ind, best_ind.fitness.values))

    import matplotlib.pyplot as plt
    plt.plot(gen, avg_list, label="average")
    plt.plot(gen, min_list, label="minimum")
    plt.plot(gen, max_list, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="upper right")
    plt.show()

main()