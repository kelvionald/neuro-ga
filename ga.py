import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import random
import math

debug = True

history_col_mutants = []
history_obj_func = []
history_epochs = []

values_neurons_num = list(range(30, 50)) if debug else list(range(30, 100))
values_layers_num = list(range(1, 3)) if debug else list(range(1, 5))
values_activation_type = [
    'relu',
    # 'tanh',
    # 'elu',
    # 'selu',
    # 'sigmoid',
    # 'exponential'
]
values_epochs_num = list(range(1, 3)) if debug else list(range(1, 5))

batch_size = 128
iter_epochs = 10
len_population = 10
kMutation = 30

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train)  # , num_classes)
y_test = keras.utils.to_categorical(y_test)  # , num_classes)


def getGen(pos):
    if pos == 0:
        return random.choice(values_neurons_num)
    elif pos == 1:
        return random.choice(values_layers_num)
    elif pos == 2:
        return random.choice(values_activation_type)
    elif pos == 3:
        return random.choice(values_epochs_num)


def getChromosome():
    return [
        getGen(0),
        getGen(1),
        getGen(2),
        getGen(3)
    ]


def genModel(finded_string):
    neurons_num, layers_num, activation_type, epochs = finded_string
    model = []
    # print([neurons_num, layers_num, activation_type])
    for i in range(0, layers_num):
        model.append(layers.Dense(neurons_num, activation=activation_type, input_shape=(784,)))
    model.append(layers.Dense(num_classes, activation="softmax"))
    model = keras.Sequential(model)
    # model.summary()
    return model


def percenify(a):
    a *= 10000
    if not math.isnan(a):
        return str(int(a) / 100) + '%'
    return a


def prtElement(text, el, score):
    neurons_num, layers_num, activation_type, epochs = el
    print(
        f'{text}: n: {neurons_num} l: {layers_num} f: {activation_type} e: {epochs} acc: {percenify(score[1])} mut: {kMutation}%')


def cacher(func):
    cache = {}

    def wrapper(*args, **kwargs):
        key = '_'.join(map(str, args[0])) + f'_{kMutation}'
        # print('KEY', key)
        if key in cache:
            return cache[key]
        cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


@cacher
def objective_function(finded_string):
    neurons_num, layers_num, activation_type, epochs = finded_string
    model = genModel(finded_string)
    # sparse_categorical_crossentropy
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    verbose = 0
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=0.1)
    score = model.evaluate(x_test, y_test, verbose=0)
    prtElement('RESULT', finded_string, score)
    print('------')
    return score[1]


def init_population(count_chromosomes=len_population):
    sum_scores = 0
    new_chromosomes = []
    while (True):
        new_chromosome = getChromosome()

        if objective_function(new_chromosome):
            new_chromosomes.append(new_chromosome)

        if count_chromosomes == len(new_chromosomes):
            break

    return new_chromosomes


def calculate_score(chromosomes):
    sum_score = 0
    for chromosome in chromosomes:
        sum_score += objective_function(chromosome)
    return sum_score


def operator_roulette(chromosomes=[]):
    rotation_roulette = []
    selected_chromosomes = []
    score_chromosomes = []

    sum_scores = calculate_score(chromosomes)
    prev_score = 0
    rotation_roulette = np.random.randint(1, 10, size=len(chromosomes)) / 10.0

    for chromosome in chromosomes:
        score_chromosomes.append(objective_function(chromosome))

    history_obj_func.append(round(sum(score_chromosomes) / len(chromosomes), 2))
    roulette_sector = []
    for chromosome, score in zip(chromosomes, score_chromosomes):
        roulette_sector.append((np.round(prev_score, 2), np.round((score / sum_scores) + prev_score, 2)))
        prev_score += np.float(score / sum_scores)

    for chromosome, score_chromosome in zip(chromosomes, roulette_sector):
        for selected_sector in rotation_roulette:
            if score_chromosome[0] < selected_sector <= score_chromosome[1]:
                selected_chromosomes.append(chromosome)

    return selected_chromosomes


def operator_crossingover(population):
    len_population = len(population)
    slice_population = int(len_population / 2)
    new_population = []

    for husband, wife in zip(population[:slice_population], population[slice_population:len_population]):
        children1 = []
        children2 = []
        crossing = np.random.randint(1, len(husband))
        children1 = husband[:crossing] + wife[crossing:]
        new_population.append(children1)
        children2 = wife[:crossing] + husband[crossing:]
        new_population.append(children2)
    return new_population


def operator_mutation(chromosomes):
    new_chromosomes = []
    count_mutants = 0
    for chromosome in chromosomes:
        if np.random.randint(0, 100) <= kMutation:
            count_mutants += 1
            len_cromosome = len(chromosome)
            crossing = np.random.randint(0, len_cromosome)
            mutation_gen = getGen(crossing)
            chromosome[crossing] = mutation_gen
            new_chromosomes.append(chromosome)
        else:
            new_chromosomes.append(chromosome)
    # history_col_mutants.append( round( count_mutants / len( chromosomes ), 2) )
    history_col_mutants.append(count_mutants)
    return new_chromosomes


def operator_selection(chromosomes):
    result = []
    new_chromosomes = []
    score = 0
    for chromosome in chromosomes:
        score = objective_function(chromosome)
        new_chromosomes.append((score, chromosome))
    new_chromosomes = sorted(new_chromosomes, reverse=True)
    for key, val in new_chromosomes:
        result.append(val)
    return result[:int(len(chromosomes) / 2)]