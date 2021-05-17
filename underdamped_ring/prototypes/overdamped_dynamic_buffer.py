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
batch_size = 32
reward_learning_rate = 10**(-2) / batch_size
final_learning_rate = 10**(-3) / batch_size
training_steps = 300
save_rate = 1
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
parameter_number = 3
force_learning_rate = 10**(1) / batch_size
value_learning_rate = 10**(1) / batch_size
bias = -0.3
replays_saved = 64
replay_buffer = np.zeros(replays_saved)
replacement_rate = 1

basis_index = np.arange(parameter_number) + 1
padding_array = np.array([1 for i in range(batch_size)])[:,np.newaxis]
force_parameters = np.zeros(parameter_number * 2 + 1)
force_parameters[0] = 1
force_parameters[1] = 2
force_current_update = np.zeros(parameter_number * 2 + 1)
value_parameters = np.zeros(parameter_number * 2 + 1)
value_current_update = np.zeros(parameter_number * 2 + 1)
features = np.zeros(parameter_number * 2 + 1)
features[0] = 1
previous_features = np.zeros(parameter_number * 2 + 1)

current_force = 0
current_value = 0
previous_value = 0


@numba.njit
def force(positions):
	return magnitude * np.sin(positions) + driving

@numba.njit
def fourier_basis(positions):
	trig_arguments = np.outer(positions, basis_index)
	sin_basis = np.sin(trig_arguments)
	cos_basis = np.cos(trig_arguments)
	return np.concatenate((padding_array, sin_basis, cos_basis), axis = 1)

@numba.njit
def fourier_basis_single(position):
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
def regularized_rewards(positions, position_changes, forces):
	reward = (position_changes - time_step * forces)**2/divisor	
	reward -= (position_changes - time_step * force(positions))**2/divisor
	reward -= bias * position_changes
	return reward

@numba.njit
def fill_buffer(position, replay_buffer, spacing):
	for index in range(replays_saved):
		for steps in range(spacing):
			unit_noise = numpy.random.randn()
			position_change = variance * unit_noise
			position += position_change
			position -= 2*math.pi*math.floor(position / (2*math.pi))
		replay_buffer[index] = position

@numba.njit
def train(steps, save_period, average_reward, force_parameters,
		  value_parameters, reward_learning_rate, replay_buffer):
	rewards = np.zeros(int(steps / save_period))
	for step in range(steps):
		batch_index = numpy.random.randint(0, replays_saved - 1, batch_size)
		positions = replay_buffer[batch_index]
		previous_features = fourier_basis(positions)
		previous_values = previous_features @ value_parameters
		parameterized_forces = previous_features @ force_parameters
		unit_noise = numpy.random.randn(batch_size)
		noise = variance * unit_noise
		position_changes = time_step * parameterized_forces + noise
		reward = regularized_rewards(positions, position_changes, parameterized_forces)
		positions += position_changes
		positions -= 2*math.pi*np.floor(positions / (2*math.pi))
		next_features = fourier_basis(positions)
		next_values = next_features @ value_parameters
		td_errors = next_values + reward - average_reward - previous_values
		value_parameters += value_learning_rate * td_errors @ previous_features
		update = (td_errors * unit_noise) @ previous_features
		force_parameters += force_learning_rate * update
		average_reward += reward_learning_rate * np.sum(td_errors)
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

fill_buffer(0, replay_buffer, 1000)
#batch = gen.choice(replay_buffer, batch_size)
#print(batch)
#print(batch[:,0])
#features = fourier_basis(batch[:,0])
#print(features)
#print(features @ value_parameters)

rewards = train(100, 10, average_reward, force_parameters, value_parameters, 
				reward_learning_rate, replay_buffer)
initial_time = time.time()
rewards = train(training_steps, save_rate, average_reward, force_parameters, 
				value_parameters, reward_learning_rate, replay_buffer)
print(time.time() - initial_time)
print(rewards[-1])
print(force_parameters)

plt.plot(rewards)
plt.show()