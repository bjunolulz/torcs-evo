import tensorflow as tf
import numpy as np
import pygad.kerasga
import pygad
from gym_torcs import TorcsEnv
import math
import sys
import csv
import copy

VISION = False

def fitness_func(ga_instance, solution, sol_idx):
    global keras_ga, model, env, flag, flag2, write

    if flag == ga_instance.generations_completed: #if new generation print the best and save it, also relaunch torcs because of memory leak
        flag += 1
        best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=best_solution_idx))
        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=best_solution)
        model.set_weights(weights=model_weights_matrix)
        model.save("torcs_weights.h5")
        print("Fitness of the generation {flag3}: {fit}".format(fit=ga_instance.last_generation_fitness, flag3= flag))
        #print(ga_instance.last_generation_fitness)
        sys.stdout.flush()
        ob = env.reset(relaunch=True)
    
    else:
        ob = env.reset()

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)
    
    sum_reward = 0
    done = False
    while (not done):
        state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        q_values = model.predict(state.reshape(1, state.shape[0]))
        action = np.zeros([1,3])
        action[0][0] = q_values[0][0]
        action[0][1] = q_values[0][1]
        action[0][2] = q_values[0][2]
        #print(action[0][0], action[0][1], action[0][2])
        observation_next, reward, done, trunc = env.step(action[0])
        ob = observation_next
        sum_reward += reward
        #print(sum_reward)
        if math.isnan(sum_reward):
            return -10000
            #ob = env.reset()

    #print(sum_reward, ga_instance.generations_completed, sol_idx)
    sys.stdout.flush()
    
    return sum_reward


def play_game(env, model):
    model.load_weights("CGtrack3/torcs_best_weights.h5")

    ob = env.reset()
    while True:
        state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        q_values = model.predict(state.reshape(1, state.shape[0]))
        action = np.zeros([1,3])
        action[0][0] = q_values[0][0]
        action[0][1] = q_values[0][1]
        action[0][2] = q_values[0][2]
        #print(action[0][0], action[0][1], action[0][2])
        observation_next, reward, done, trunc = env.step(action[0])
        ob = observation_next


def create_model():
    S = tf.keras.layers.Input(shape=(29, ))
    h0 = tf.keras.layers.Dense(100, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-4), activation=lambda x: tf.keras.layers.LeakyReLU(alpha=0.3)(x), kernel_regularizer=tf.keras.regularizers.l2(0.01))(S)
    drop0 = tf.keras.layers.Dropout(0.5)(h0) # add dropout
    #n1 = tf.keras.layers.Normalization(axis=None, mean=0.0, variance=1e-2)(drop0) # change h0 to drop0
    h1 = tf.keras.layers.Dense(200, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-4), activation=lambda x: tf.keras.layers.LeakyReLU(alpha=0.3)(x), kernel_regularizer=tf.keras.regularizers.l2(0.01))(drop0)
    drop1 = tf.keras.layers.Dropout(0.5)(h1) # add dropout
    #n2 = tf.keras.layers.Normalization(axis=None, mean=0.0, variance=1e-2)(drop1) # change h1 to drop1
    h2 = tf.keras.layers.Dense(50, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-4), activation=lambda x: tf.keras.layers.LeakyReLU(alpha=0.3)(x), kernel_regularizer=tf.keras.regularizers.l2(0.01))(drop1)
    drop2 = tf.keras.layers.Dropout(0.5)(h2) # add dropout
    #n3 = tf.keras.layers.Normalization(axis=None, mean=0.0, variance=1e-2)(drop2) # change h2 to drop2

    Steering = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-3), activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.01))(drop2)
    Acceleration = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-3), activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(drop2)
    Brake = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-3), activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(drop2)

    V = tf.keras.layers.Concatenate()([Steering, Acceleration, Brake])
    model = tf.keras.Model(inputs=S, outputs=V)
    return model

#write output to file 
sys.stdout = open('out.log', 'w')
sys.stderr = sys.stdout

fitness = open('fitness.csv', 'w')
write = csv.writer(fitness)

#import env
env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

flag = 0
flag2 = 0

#kernel_initializer=tf.keras.initializers.HeNormal(seed=None)

#create model

model = create_model()

model.summary()
# model.build(input_shape=(29,))
# print(model.get_weights())

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)

init_pop = []
for i in range(keras_ga.num_solutions):
    tmp_model = create_model()
    tmp_model.compile(loss='mean_absolute_error', optimizer='adam')
    init_pop.append(pygad.kerasga.model_weights_as_vector(tmp_model))
    print(pygad.kerasga.model_weights_as_vector(tmp_model))

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=5,
                       keep_parents=-1,
                       initial_population=init_pop,
                       fitness_func=fitness_func,
                       parent_selection_type="sss",
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=50,
                       mutation_probability=0.3,
                       mutation_by_replacement=False,
                       random_mutation_min_val=-0.1,
                       random_mutation_max_val=0.1,
                       save_solutions=True)

#train or play
train = False
if train:
    ga_instance.run()
    last_gen = ga_instance.solutions[-10:]
    c = 0
    for g in last_gen:
        m = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=g)
        model.set_weights(weights=m)
        model.save("{ca}.h5".format(ca=c))
        c += 1

    #ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)
    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    for solu in ga_instance.best_solutions_fitness:
        if solu is not None:
            write.writerow([solu])

    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=best_solution_idx))
    sys.stdout.flush()
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=best_solution)
    model.set_weights(weights=model_weights_matrix)
    model.save("torcs_best_weights.h5")
else:
    play_game(env, model)

env.end()