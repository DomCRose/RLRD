import math
import copy
import time
import numba
import numpy as np
import numpy.random
from matplotlib import pyplot as plt

magnitude = 2
driving = 1
time_step = 0.001
divisor = 4 * time_step
variance = (2*time_step)**0.5
average_reward = 0
reward_learning_rate = 10**(-5)
final_learning_rate = 10**(-6)
training_steps = 2000000
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
parameter_number = 5
force_learning_rate = 10**(-1)
value_learning_rate = 10**(-1)
ss_learning_rate = 10**(-3)
bias = 1.5

basis_index = np.arange(parameter_number) + 1
force_parameters = np.zeros(parameter_number * 2 + 1)
force_current_update = np.zeros(parameter_number * 2 + 1)
value_parameters = np.zeros(parameter_number * 2 + 1)
value_current_update = np.zeros(parameter_number * 2 + 1)
steady_state_discretization = 10
steady_state_bin_width = 2*math.pi / steady_state_discretization
steady_state_parameters = np.zeros(steady_state_discretization)
features = np.zeros(parameter_number * 2 + 1)
features[0] = 1
previous_features = np.zeros(parameter_number * 2 + 1)

current_force = 0
current_value = 0
previous_value = 0


@numba.njit
def force(position):
	return magnitude * math.sin(position) + driving

@numba.njit
def fourier_basis(position):
	trig_arguments = position * basis_index
	sin_basis = np.sin(trig_arguments)
	cos_basis = np.cos(trig_arguments)
	return np.concatenate((np.array([1]), sin_basis, cos_basis))

@numba.njit
def trajectory(position, length):
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
	reward = noise**2/divisor	
	reward -= (position_change - time_step * force(position))**2/divisor
	reward -= bias * position_change
	return reward

@numba.njit
def train(position, steps, save_period, average_reward, force_parameters,
		  value_parameters, steady_state_parameters, reward_learning_rate, bias):
	rewards = np.zeros(int(steps / save_period))
	features = fourier_basis(position)
	current_value = value_parameters @ features
	positions = [position]
	for step in range(steps):
		previous_value = current_value
		previous_features = np.copy(features)
		unit_noise = numpy.random.randn()
		noise = variance * unit_noise
		parameterized_force = force_parameters @ features
		position_change = time_step * parameterized_force + noise
		reward = regularized_reward(position, position_change, noise, bias)
		position += position_change
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		positions.append(position)
		features = fourier_basis(position)
		current_value = value_parameters @ features
		pos_index = math.floor(position / steady_state_bin_width)
		ss_norm = np.sum(np.exp(steady_state_parameters))
		probability = np.exp(steady_state_parameters[pos_index]) / ss_norm
		td_error = current_value + reward - average_reward - previous_value
		force_parameters += (force_learning_rate * td_error * unit_noise
							 * previous_features)
		value_parameters += value_learning_rate * td_error * previous_features
		average_reward += reward_learning_rate * td_error
		reward_learning_rate += linear_rate_change
		steady_state_parameters[pos_index] += ss_learning_rate * (1 - probability)
		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
	return rewards, average_reward, force_parameters, value_parameters, steady_state_parameters, positions

rewards, average_reward, force_parameters, value_parameters, steady_state_parameters, positions = train(0, training_steps, 10, average_reward, force_parameters, 
	  value_parameters, steady_state_parameters, reward_learning_rate, bias)
	
ss_norm = np.sum(np.exp(steady_state_parameters))
probabilities = np.exp(steady_state_parameters) / ss_norm

plt.plot(probabilities)
plt.show()

plt.plot(positions)
plt.show()