import gym
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pygad.kerasga
import pygad
from gym_torcs import TorcsEnv
import math

VISION = False

def fitness_func(self, solution, sol_idx):
    global keras_ga, model, observation_space_size, env, ga_instance, flag

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)

    # play game
    if flag == ga_instance.generations_completed:
        flag +=1
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
        model.set_weights(weights=model_weights_matrix)
        model.save("torcs_weights")
        ob = env.reset(relaunch = True)

    else:
        ob = env.reset()
    sum_reward = 0
    done = False
    c = 0
    while (not done) and c<1000:
        #state = np.reshape(observation, [1, observation_space_size])
        state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        q_values = model.predict(state.reshape(1, state.shape[0]))
        action = np.zeros([1,3])
        action[0][0] = q_values[0][0]
        action[0][1] = q_values[0][1]
        #action[0][2] = q_values[0][2]
        observation_next, reward, done, trunc = env.step(action[0])
        ob = observation_next
        sum_reward += reward
        c += 1
        #print(sum_reward)
        if math.isnan(sum_reward):
            ob = env.reset(relaunch = True)

    return sum_reward


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)
observation_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.shape[0]

flag = 0

model = Sequential()
model.add(Dense(16, input_shape=(None, 29), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(action_space_size, activation='linear'))
model.summary()

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)

ga_instance = pygad.GA(num_generations=500,
                       num_parents_mating=5,
                       initial_population=keras_ga.population_weights,
                       fitness_func=fitness_func,
                       parent_selection_type="sss",
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10,
                       keep_parents=-1,
                       on_generation=callback_generation)

ga_instance.run()

ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
model.set_weights(weights=model_weights_matrix)
model.save("torcs_weights")