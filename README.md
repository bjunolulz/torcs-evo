# torcs-evo
This is a project with the goal of making an agent for torcs using the genetic algorithm. It is made using Keras and PyGAD.

## How to use?
1. Install Python and pip if you already don't have it
2. Install Tensorflow
3. Install PyGAD
4. Download gym-torcs (https://github.com/ugo-nama-kun/gym_torcs) and follow the install instructions there
5. To train set the train flag in initial_gen.py and next_gens.py to True
6. Run by running python3 initial_gen.py or python3 next_gens.py
7. To run the learned agent change the path in the play_game method and set the train flag to False

## Exclaimer
Runs only on Linux, I have not tested it on Windows, maybe try your luck with WSL. There are two files, initial_gen.py and next_gen.py. Use initial_gen.py when starting training for x number of generations, and then continue with next_gens.py to avoid crashing.

## Problems
I won't be adding work to this project anymore, but the problems I encountered are the following:
1. RAM - This sucks RAM and there is a memory leak somewhere. I don't know if it is just Tensorflow or what, but for this reason the code is split into two files. I can run initial_gen.py for 100 generations without crashing and then saving the current population and running next_gens.py for another 50. You can run next_gens.py as many times as you want.
2. The neural network has problems with saturation. I have managed to fix it somewhat with kernel initialization but it is still not what I wanted it to be. It cannot drive straight on the road but is pretty good in managing curves. EDIT: Try playing around with the fitness function, try removing the subtraction of the distance from the middle of the track.

## Related work
This project (https://github.com/yanpanlau/DDPG-Keras-Torcs) was used as an inspiration so go look there if you want any more information.
