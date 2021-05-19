# Reinforcement Learning for Rare Diffusive Dynamics
 
This repository contains the code used to generate the results found in the corresponding [paper](https://arxiv.org/abs/2105.04321).

Each folder is named after the model it contains the code for.

In finite time, where we learn a time-dependent dynamics producing constrained trajectories, this includes:
+ **Brownian bridges:** Brownian trajectories beginning at 0 and ending at 1.
+ **Multiple pathways:** A potential with two degenerate classes of barrier-crossing trajectories.
+ **Mueller-Brown:** A potential with steep barriers and two minima connected by a shallow metastable intermediate well. We study diffusive trajectories going across the metastable intermediate well to go from a local to a deep global minimum.

In infinite time, where we learn a time-homogenous dynamics in the stationary state, we consider a driven particle in a periodic potential on a ring, studying the statistics of its time-integrated velocity. We consider two cases:
+ **Overdamped:** Here the dynamical system consists of only the particle position, evolving according to a drift force and noise.
+ **Underdamped:** Here the dynamical system consists of the position evolving according to a velocity, where the velocity evolves with a drift force and noise.

Data produced using this code is available at Zenodo (add link when available).
