import gym
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, Input, Concatenate
import pygad.kerasga
import pygad
from gym_torcs import TorcsEnv
import math
import sys
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow import random as ra
import random
import time

VISION = False

def fitness_func(solution, sol_idx):
    global keras_ga, model, env, flag, flag2

    if flag == ga_instance.generations_completed: #if new generation print the best and save it, also relaunch torcs because of memory leak
        flag += 1
        best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=best_solution_idx))

        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=best_solution)
        model.set_weights(weights=model_weights_matrix)
        model.save("torcs_weights")
        print(ga_instance.last_generation_fitness)
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
    model.load_weights("torcs_weights")

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
        

# def callback(ga_instance):
#     env.reset_torcs()

# def callback_generation(ga_instance):
#     print("Generation = {generation}".format(generation=ga_instance.generations_completed))
#     sys.stdout.flush()
#     print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
#     sys.stdout.flush()
#     best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
#     sys.stdout.flush()
#     model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=best_solution)
#     model.set_weights(weights=model_weights_matrix)
#     model.save("torcs_weights") #save the best from the generation
#     ob = env.reset(relaunch=True)

#write output to file
sys.stdout = open('out.log', 'w')
sys.stderr = sys.stdout

#import env
env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

flag = 0
flag2 = 0

#create model
S = Input(shape=(29, ))   
h0 = Dense(25, activation='relu')(S)
h1 = Dense(25, activation='relu')(h0)
Steering = Dense(1,activation='tanh', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=time.time()))(h1)
Brake = Dense(1,activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=time.time()))(h1)
Acceleration = Dense(1,activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=time.time()))(h1)
V = Concatenate()([Steering, Acceleration, Brake])
model = Model(inputs=S, outputs=V)

model.summary()

#create keras model
keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=5,
                       keep_parents=-1,
                       initial_population=keras_ga.population_weights,
                       fitness_func=fitness_func,
                       parent_selection_type="sss",
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10,
                       mutation_probability=0.3)

#train or play
train = True
if train:
    ga_instance.run()

    ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    sys.stdout.flush()
    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)
    model.save("torcs_weights")
    ga_instance.save(filename='ga')
else:
    play_game(env, model)

env.end()