# Pathfinding on Cayley graphs using diffusion models


This repo contains code for training and evaluating diffusion models for pathfinding on Cayley graphs. The simplest cases are the general permutation group, and the 2x2x2 and 3x3x3 cube groups. This is the repository attached to the paper "Diffusion Models for Cayley Graphs" (ADV. THEOR. MATH. PHYS. 2025).

In the past, people have proposed learning the inverse of scrambling trajectories from the solved state solve the Rubik's cube. We simply propose taking that idea  seriously, and thereby consider the problem as a diffusion modelling exercise. The scrambling defines a forward diffusion process starting from the solved configuration, which is a time-homogeneous discrete-time Markov chain. It has an inverse which can be written analytically and is also a Markov chain, but time-inhomogeneous. If the transition kernel for the forward process is $q_t(x_{t+1}|x_t)=q(x_{t+1}|x_t)$ then the transition kernel for the inverse process is $\tilde{q}_t(x_{t-1}|x_t)=q(x_t|x_{t-1})\frac{p_{t-1}(x_{t-1})}{p_t(x_t)}$ where $p_t$ is the distribution from the forward process at time $t$.

Minimising a suitable choice of score-matching objective (a Bregman divergence), the model learns the inverse of the scrambling process. Running the inverse diffusive process, in principle we solve the pathfinding problem.

![Comparison](images/Comparison.png)

This method is considerably more sample efficient (by more than an order of magnitude) than the previous best method, EfficientCube. EfficientCube uses a neural network to directly predict the inverse of each scrambling move. Our work places this on a firmer ground, which can easily be seen by considering the long-time limit of EfficientCube. If one were to train with $T$ to $\infty$, just learning the inverse of each scrambling move would be meaningless, as the stationary distribution after time $T$ would be the uniform distribution. If one instead learns the inverse of the diffusive process, the network would be able to distinguish between the early and late stages of the scrambling process. Theoretically, therefore, we *remove* a hyperparameter from the training process, as we could use any $T$ that is sufficiently large. In practice, of course, we need to train with a finite $T$, but this is still a significant improvement.

# Applications to the Andrews-Curtis conjecture (*in progress*)

We can apply the same methods to the Andrews-Curtis conjecture. Starting with a presentation of the trivial group, apply random Andrews-Curtis moves of type AC1' and AC2' (see https://arxiv.org/abs/2408.15332). Then learn to invert this diffusive process.

We restrict to balanced presentations with two relators and two generators.

We are able to trivialise AK(2) using this method. We are not able to trivialise AK(3), so far. AK(3) is the first element in the Akbulut-Kirby series which is not known to be trivial (or, indeed, stably trivial).

# NN models implemented:

- `FF_residual_model`: A feedforward residual network with a time embedding.
- `Transformer_model`: A transformer model with a time embedding.

# Beam search

`heuristic_searches.py` contains an efficient implementation of beam search, with preallocation of memory for the beam tensors. It represents a 10x speeup over other implementations for the Rubik's cube.


# Other groups
The code is designed to be modular, so that it can be easily extended to other groups.

The main files are:

- `envsAndScramble.py`: contains the environment definitions and scrambling functions for the cube groups.
- `NN_models.py`: contains the definition of the neural network models.
- `cube_training.py`: contains the training loop for the cube groups.
- `heuristic_searches.py`: contains the heuristic search methods.


# Troubleshoot:

Ensure you install the correct version of torch-scatter.
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
```

Ensure you have installed wandb, pytorch_optimizer, and tqdm.

# Credit
Rubik's cube environment, and statistical comparison data taken from EfficientCube by K. Takano.

# MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.