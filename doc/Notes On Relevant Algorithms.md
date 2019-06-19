Notes On Neural Networks
------------------------

Structure
---------------
- The input layer shall consist of only one neuron
- There will be two middle layers of the same size with a comparatively large number of
  neurons
- The output layer shall be of the same size as the test video data

Weights
-------

- Weights will be in the range of -1 to 1.
         - Using ReLu allows the output of the hidden layers to be greater than zero
           which will be necessary for the specialized output layer function
- Weights will be initialized to very small positive values close to zero

Bias
----

- Bias shall be initialized to very small positive values close to zero
- Bias shall be kept as a single number and added to the layer through addition
- Bias shall be added to each of the neurons before applying the activation function

Activation functions
--------------------

- The activation functions of the hidden layers shall be the ReLu function:
      
      : max(0, x)

- The activation function shall be a specialized function that scales a float in the range of 0 to 1 to an integer in the range of 0 to 255

---------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------

Notes on Genetic Algorithms
---------------------------

A brief overview of the Genetic Algorithm.

Here is the pseudocode for a Genetic Algorithm:

1. Initialize Population
2. Rate entire population
3. Select two members of the population using Roulette Wheel Selection
4. Crossover the two members of the population according to the crossover rate
5. Mutate the two members
6. Repeat steps 3 through 5 until a new population of N members has been produced

Behaviors
---------

- Weights will be mutated by randomly picking either positive or negative and
  bumping them in that direction by a very small margin.
