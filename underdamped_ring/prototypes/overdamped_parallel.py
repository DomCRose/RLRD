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
trajectory_number = 64
reward_learning_rate = 10**(-3)/trajectory_number
final_learning_rate = 10**(-3)/trajectory_number
training_steps = 100000
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
parameter_number = 5
force_learning_rate = 10**(-1)/trajectory_number
value_learning_rate = 10**(-0.5)/trajectory_number
bias = 1.3

basis_index = np.arange(parameter_number) + 1
force_parameters = np.zeros(parameter_number * 2 + 1)
force_parameters[0] = 1
force_parameters[1] = 2
force_current_update = np.zeros(parameter_number * 2 + 1)
value_parameters = np.zeros(parameter_number * 2 + 1)
value_current_update = np.zeros(parameter_number * 2 + 1)
features = np.zeros(parameter_number * 2 + 1)
features[0] = 1
previous_features = np.zeros(parameter_number * 2 + 1)
padding_array = np.array([1 for i in range(trajectory_number)])[:,np.newaxis]

current_force = 0
current_value = 0
previous_value = 0


@numba.njit
def force(position):
	return magnitude * np.sin(position) + driving

@numba.njit
def fourier_basis(position):
	trig_arguments = np.outer(position, basis_index)
	sin_basis = np.sin(trig_arguments)
	cos_basis = np.cos(trig_arguments)
	return np.concatenate((padding_array, sin_basis, cos_basis), axis = 1)

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
	#print(noise.shape)
	#print(position_change.shape)
	#print(force(position).shape)
	reward = noise**2/divisor	
	reward -= (position_change - time_step * force(position))**2/divisor
	reward -= bias * position_change
	return reward

@numba.njit
def instant_regularized_reward(position, parameterized_force, bias):
	reward = -time_step * (parameterized_force - force(position))**2/divisor
	#print(reward)
	reward -= bias * time_step * parameterized_force
	#print('KL')
	#print(-time_step * (parameterized_force - force(position))**2/divisor)
	#print('Bias')
	#print(-bias * time_step * parameterized_force)
	#print('Reward')
	#print(reward)
	return reward

@numba.njit
def train(positions, steps, save_period, average_reward, force_parameters,
		  value_parameters, reward_learning_rate, bias):
	rewards = np.zeros(int(steps / save_period))
	features = fourier_basis(positions)
	current_value = features @ value_parameters
	for step in range(steps):
		previous_value = current_value
		previous_features = np.copy(features)
		unit_noise = numpy.random.randn(trajectory_number)
		noise = variance * unit_noise
		parameterized_force = features @ force_parameters
		position_change = time_step * parameterized_force + noise
		reward = regularized_reward(positions, position_change, noise, bias)
		#reward = instant_regularized_reward(positions, parameterized_force, bias)
		positions += position_change
		positions -= 2*math.pi*np.floor(positions / (2*math.pi))
		features = fourier_basis(positions)
		current_value = features @ value_parameters
		td_error = current_value + reward - average_reward - previous_value
		force_parameters += (force_learning_rate * (td_error * unit_noise)
							 @ previous_features)
		value_parameters += value_learning_rate * td_error @ previous_features
		average_reward += reward_learning_rate * np.sum(td_error)
		reward_learning_rate += linear_rate_change
		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
	return rewards, average_reward, force_parameters, value_parameters

#sample_trajectory = trajectory(0, 10)
#initial_time = time.time()
#sample_trajectory = trajectory(0, 100000)
#print(time.time() - initial_time)
#plt.plot(sample_trajectory)
#plt.show()
positions = np.random.random(trajectory_number) * 2 * np.pi
print(positions)
bias_number = 20
bias_step = 2/bias_number
biases = [-0.5 + bias_step*i for i in range(bias_number + 1)]
scgf = []
learning_curves = []
for bias in biases:
	initial_time = time.time()
	rewards, average_reward, force_parameters, value_parameters = train(
		positions, training_steps, 10, average_reward, force_parameters, 
		value_parameters, reward_learning_rate, bias)
	print(time.time() - initial_time)
	scgf.append(rewards[-1])
	learning_curves.append(rewards)

fig = plt.figure(figsize = (10,4))
plt.subplot(121)
plt.plot(biases, scgf)
plt.xlabel(r'$s$')
plt.ylabel(r'$\theta(s)$')
plt.subplot(122)
for curve in learning_curves:
	plt.plot(curve)
plt.xlabel('training steps')
plt.ylabel(r'$\theta(s)$')

#fig.savefig("overdamped_2000000steps.png", dpi = 500)
plt.show()