"""
Written by Dominic C. Rose, May 2021, dominicdrose@gmail.com

Optimizes a force for the study of the time-integrated velocity of a particle on a ring,
with a periodic potential and a constant driving force in the overdamped case.

Uses an Actor-Critic algorithm, with temporal-difference errors calculated using single
time-step segments of trajectory from a discretized diffusive simulation. Updates are done
for each individual transition.
"""
import math
import copy
import time
import numba
import numpy as np
import numpy.random
from matplotlib import pyplot as plt

""" Dynamics parameters. """
magnitude = 2
driving = 1
time_step = 0.001
divisor = 4 * time_step
variance = (2*time_step)**0.5

""" Variable initialization. """
average_reward = 0
average_kl = 0
average_velocity = 0

""" Algorithm parameters. """
reward_learning_rate = 10**(-5)
final_learning_rate = 10**(-6)
training_steps = 10000000
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
force_learning_rate = 10**(-1)
value_learning_rate = 10**(-2)

""" Approximation parameters. """
parameter_number = 5

""" Approximation setup. """
basis_index = np.arange(parameter_number) + 1
force_parameters = np.zeros(parameter_number * 2 + 1)
force_parameters[0] = 1
force_parameters[1] = 2
force_current_update = np.zeros(parameter_number * 2 + 1)
value_parameters = np.zeros(parameter_number * 2 + 1)
value_current_update = np.zeros(parameter_number * 2 + 1)



@numba.njit
def force(position):
	"""Calculates the force applied directly to the particle position."""
	return magnitude * math.sin(position) + driving

@numba.njit
def fourier_basis(position):
	"""Calculates the values of a Fourier basis at a given position."""
	trig_arguments = position * basis_index
	sin_basis = np.sin(trig_arguments)
	cos_basis = np.cos(trig_arguments)
	return np.concatenate((np.array([1]), sin_basis, cos_basis))

@numba.njit
def trajectory(position, length):
	"""Evolves given position with the original dynamics."""
	trajectory = np.zeros(length + 1)
	trajectory[0] = position
	for i in range(1, length + 1):
		noise = variance * numpy.random.randn()
		position += time_step * force(position) + noise
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		trajectory[i] = position
	return trajectory

@numba.njit
def regularized_reward(position, position_change, noise, bias):
	"""Calculates the reward for the specified transition, using noise.
	
	Assumes the current dynamics was used to generate the transition
	with the sampled noise being that given as an input.
	"""
	reward = noise**2/divisor	
	reward -= (position_change - time_step * force(position))**2/divisor
	kl_reg = reward
	reward -= bias * position_change
	return reward, kl_reg

@numba.njit
def train(position, steps, save_period, average_reward, force_parameters,
		  value_parameters, reward_learning_rate, bias, average_kl, average_velocity):
	"""Trains the dynamics for the specified number of steps."""
	rewards = np.zeros(int(steps / save_period))
	kl_divergences = np.zeros(int(steps / save_period))
	velocities = np.zeros(int(steps / save_period))
	features = fourier_basis(position)
	current_value = value_parameters @ features
	for step in range(steps):
		previous_value = current_value
		previous_features = np.copy(features)
		unit_noise = numpy.random.randn()
		noise = variance * unit_noise
		parameterized_force = force_parameters @ features
		position_change = time_step * parameterized_force + noise
		reward, kl_reg = regularized_reward(position, position_change, noise, bias)
		position += position_change
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_basis(position)
		current_value = value_parameters @ features
		td_error = current_value + reward - average_reward - previous_value
		force_parameters += (force_learning_rate * td_error * unit_noise
							 * previous_features)
		value_parameters += value_learning_rate * td_error * previous_features
		average_reward += reward_learning_rate * td_error
		average_kl += reward_learning_rate*0.1 * (kl_reg - average_kl)
		average_velocity += reward_learning_rate*0.1 * (position_change - average_velocity)
		reward_learning_rate += linear_rate_change
		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
			kl_divergences[int(step/save_period)] = average_kl / time_step
			velocities[int(step/save_period)] = average_velocity / time_step
	return rewards, kl_divergences, velocities, force_parameters, value_parameters, average_reward, average_velocity, average_kl

""" Bias setup. """
bias_number = 20
initial_bias = -0.5
bias_range = 2
bias_step = bias_range/bias_number
biases = [round(initial_bias + bias_step*i, 3) for i in range(bias_number + 1)]
print(biases)

""" Data storage for plotting. """
rews = []
kldivs = []
velocits = []
scgfs = []
kls = []
velos = []

save = False
for bias in biases:
	initial_time = time.time()
	rewards, kl_divergences, velocities, force_parameters, value_parameters, average_reward, average_velocity, average_kl = train(
		0, training_steps, int(training_steps/10000), average_reward, 
		force_parameters, 
		value_parameters, reward_learning_rate, bias, average_kl, average_velocity)
	rews.append(rewards)
	kldivs.append(kl_divergences)
	velocits.append(velocities)
	scgfs.append(rewards[-1])
	kls.append(kl_divergences[-1])
	velos.append(velocities[-1])
	print(bias)
	print(time.time() - initial_time)
	if save:
		np.save("overdamped_bias%s_fb%s_ts%s_force_parameters"%(
			bias, parameter_number,
			time_step), force_parameters)
		np.save("overdamped_bias%s_fb%s_ts%s_value_parameters"%(
			bias, parameter_number,
			time_step), value_parameters)
		np.save("overdamped_bias%s_fb%s_ts%s_ls%s_scgf_estimate"%(
			bias, parameter_number,
			time_step, training_steps), rewards)
		np.save("overdamped_bias%s_fb%s_ts%s_ls%s_kl_estimate"%(
			bias, parameter_number,
			time_step, training_steps), kl_divergences)
		np.save("overdamped_bias%s_fb%s_ts%s_ls%s_velocity_estimate"%(
			bias, parameter_number,
			time_step, training_steps), velocities)


plt.figure(figsize = (13, 4))
plt.subplot(231)
for i in range(bias_number + 1):
	plt.plot(rews[i])
plt.subplot(232)
for i in range(bias_number + 1):
	plt.plot(-kldivs[i])
plt.subplot(233)
for i in range(bias_number + 1):
	plt.plot(velocits[i])
plt.subplot(234)
plt.plot(biases, scgfs)
plt.subplot(235)
plt.plot(biases, -np.array(kls))
plt.subplot(236)
plt.plot(biases, velos)
plt.show()