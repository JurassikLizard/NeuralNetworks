import random
import numpy as np


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def softmax(x):
    return (np.exp(x) / np.exp(x).sum())


def softmax_stable(x):
    return (np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

def gen_data(count):
    X = []
    y = []
    for i in range(count):
        data = []
        for i in range(6):
            data.append(random.randint(0, 10))
        X.append(data)
        if data[0:3] > data[3:6]:
            y.append([1, 0])
        else:
            y.append([0, 1])
    return X, y


def execute(pop_size, generations, threshold, network, X, y, data_func=None):
    agents = generate_agents(pop_size, network)
    for i in range(generations):
        print('Generation', str(i), ':')
        if data_func is not None:
            X, y = data_func(10)
        agents = fitness(agents, X, y)
        agents = selection(agents)
        agents = crossover(agents, network, pop_size)
        agents = mutation(agents)
        agents = fitness(agents, X, y)
        print(agents[0].fitness)
        
        if agents[0].fitness < threshold:
            print('Threshold met at generation ' + str(i) + ' !')
            print(X[0], " -\n", y[0])
            print(agents[0].neural_network.propagate(X[0]))
            print(agents[0].fitness)
            break
    
    return agents


class Agent:
    class NeuralNetwork:
        def __init__(self, network):
            self.weights = []
            self.activations = []
            for layer in network:
                if layer[0] is not None:
                    input_size = layer[0]
                else:
                    input_size = network[network.index(layer) - 1][1]
                output_size = layer[1]
                activation = layer[2]
                self.weights.append(np.random.randn(output_size, input_size)) # [layer][neuron][connection]
                self.activations.append(activation)
        
        def propagate(self, data):
            input_data = data
            for i in range(len(self.weights)):
                layer = self.weights[i]
                outputs = []
                for neuron in layer:
                    z = np.dot(input_data, neuron)
                    a = z
                    if self.activations[i] is not None:
                        a = self.activations[i](z)
                    outputs.append(a)
                if self.activations[i] is None:
                    outputs = softmax_stable(np.array(outputs))
                input_data = outputs
            yhat = input_data
            return yhat
    
    def __init__(self, network):
        self.neural_network = Agent.NeuralNetwork(network)
        self.fitness = 0


def generate_agents(population, network):
    return [Agent(network) for _ in range(population)]


def fitness(agents, X, y):
    for agent in agents:
        costs = []
        for i in range(len(X)):
            yhat = agent.neural_network.propagate(X[i])
            answer = y[i]
            cost = 0
            for j in range(len(yhat)):
                cost += (abs(answer[j] - yhat[j]) * 2) ** 2
            costs.append(cost)
        agent.fitness = sum(costs)
    return agents


def selection(agents):
    agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
    agents = agents[:int(0.2 * len(agents))]
    return agents


def unflatten(flattened, shapes):
    newarray = []
    index = 0
    for shape in shapes:
        size = np.product(shape)
        newarray.append(flattened[index: index + size].reshape(shape))
        index += size
    return newarray


def crossover(agents, network, pop_size):
    offspring = []
    for _ in range((pop_size - len(agents)) // 2):
        parent1 = random.choice(agents)
        parent2 = random.choice(agents)
        child1 = Agent(network)
        child2 = Agent(network)
        
        shapes = [a.shape for a in parent1.neural_network.weights]
        
        genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.weights])
        genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.weights])
        
        split = random.randint(0, len(genes1) - 1)
        child1_genes = np.asarray(genes1[0:split].tolist() + genes2[split:].tolist())
        child2_genes = np.asarray(genes1[0:split].tolist() + genes2[split:].tolist())
        
        child1.neural_network.weights = unflatten(child1_genes, shapes)
        child2.neural_network.weights = unflatten(child2_genes, shapes)
        
        offspring.append(child1)
        offspring.append(child2)
    agents.extend(offspring)
    return agents


def mutation(agents):
    for agent in agents:
        if random.uniform(0.0, 1.0) <= 0.1:
            weights = agent.neural_network.weights
            shapes = [a.shape for a in weights]
            flattened = np.concatenate([a.flatten() for a in weights])
            randint = random.randint(0, len(flattened) - 1)
            flattened[randint] = np.random.randn()
            new_weights = []
            weights_idx = 0
            for shape in shapes:
                size = np.product(shape)
                new_weights.append(flattened[weights_idx: weights_idx + size].reshape(shape))
                weights_idx += size
            agent.neural_network.weights = new_weights
    return agents


if __name__ == "__main__":
    random.seed(random.randint(0, 25))
    
    X, y = gen_data(10)
    
    network = [[6, 12, sigmoid], [None, 12, sigmoid], [None, 12, sigmoid], [None, 2, None]]
    
    agents = execute(100, 5000, 0.1, network, X, y) # , data_func=gen_data
