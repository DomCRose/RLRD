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
reward_learning_rate = 10**(-4)
final_learning_rate = 10**(-4)
target_change_rate = 0.5
training_steps = 200000
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
parameter_number = 5
force_learning_rate = 10**(-0)
value_learning_rate = 10**(-1)
bias = -0.3

basis_index = np.arange(parameter_number) + 1
force_parameters = np.zeros(parameter_number * 2 + 1)
force_current_update = np.zeros(parameter_number * 2 + 1)
target_value_parameters = np.zeros(parameter_number * 2 + 1)
value_parameters = np.zeros(parameter_number * 2 + 1)
value_current_update = np.zeros(parameter_number * 2 + 1)
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
def regularized_reward(position, position_change, noise):
	reward = noise**2/divisor	
	reward -= (position_change - time_step * force(position))**2/divisor
	reward -= bias * position_change
	return reward

@numba.njit
def train(position, steps, save_period, average_reward, force_parameters,
		  value_parameters, target_value_parameters, reward_learning_rate):
	rewards = np.zeros(int(steps / save_period))
	features = fourier_basis(position)
	current_value = target_value_parameters @ features
	for step in range(steps):
		previous_value = current_value
		previous_features = np.copy(features)
		unit_noise = numpy.random.randn()
		noise = variance * unit_noise
		parameterized_force = force_parameters @ features
		position_change = time_step * parameterized_force + noise
		reward = regularized_reward(position, position_change, noise)
		position += position_change
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_basis(position)
		current_value = target_value_parameters @ features
		td_error = current_value + reward - average_reward - previous_value
		force_parameters += (force_learning_rate * td_error * unit_noise
							 * previous_features)
		value_parameters += value_learning_rate * td_error * previous_features
		target_value_parameters += target_change_rate * (
			value_parameters - target_value_parameters)
		average_reward += reward_learning_rate * td_error
		reward_learning_rate += linear_rate_change
		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
	return rewards

#sample_trajectory = trajectory(0, 10)
#initial_time = time.time()
#sample_trajectory = trajectory(0, 100000)
#print(time.time() - initial_time)
#plt.plot(sample_trajectory)
#plt.show()

rewards = train(0, 100, 10, average_reward, force_parameters, value_parameters, 
				target_value_parameters, reward_learning_rate)
initial_time = time.time()
rewards = train(0, training_steps, 1000, average_reward, force_parameters, 
				value_parameters, target_value_parameters, reward_learning_rate)
print(time.time() - initial_time)

plt.plot(rewards)
plt.show()