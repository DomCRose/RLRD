# Reinforcement Learning for Rare Diffusive Dynamics
 
This repository contains the code used to generate the results found in the corresponding [paper](https://arxiv.org/abs/2105.04321).

Each folder is named after the model it contains the code for.

In finite time, this includes:
+ **Brownian bridges:** Brownian trajectories beginning at zero and ending and one.
+ **Multiple pathways:** A potential two global minima and two identical minimum-energy pathways. We study trajectories between these two global minima.
+ **Mueller-Brown:** A potential with two deep local minima and a third shallow local minimum between them. We study paths from the second deepest local minimum to the deepest local minimum, which pass through the shallow minimum and are approximately instantonic.

In infinite time, where we learn a time-homogenous dynamics in the stationary state, we consider a driven particle in a periodic potential on a ring, studying the statistics of its time-integrated velocity. We consider two cases:
+ **Overdamped:** Here the dynamical system consists of only the particles position, evolving according to a drift force and noise.
+ **Underdamped:** Here the dynamical system consists of the position evolving according to a velocity, where the velocity evolves according to a drift force and noise.

Data produced using this code is available at Zenodo (add link when available).