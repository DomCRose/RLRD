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
batch_size = 64
reward_learning_rate = 10**(-3) / batch_size
final_learning_rate = 10**(-3) / batch_size
training_steps = 10000000
save_rate = 1
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
parameter_number = 3
force_learning_rate = 10**(-4) / batch_size
value_learning_rate = 10**(-5) / batch_size
bias = 1.3
replays_saved = 1000000
replay_buffer = np.zeros((replays_saved,4))
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
def regularized_rewards(positions, features, position_changes, force_parameters):
	reward = (position_changes - time_step * features @ force_parameters)**2/divisor	
	reward -= (position_changes - time_step * force(positions))**2/divisor
	reward -= bias * position_changes
	return reward

@numba.njit
def instant_regularized_reward(position, parameterized_force):
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
def fill_buffer(position, replay_buffer, spacing):
	for replay in replay_buffer:
		for steps in range(spacing):
			unit_noise = numpy.random.randn()
			position_change = variance * unit_noise
			position += position_change
			position -= 2*math.pi*math.floor(position / (2*math.pi))
		unit_noise = numpy.random.randn()
		position_change = variance * unit_noise
		replay[0] = position
		position += position_change
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		replay[1] = position
		replay[2] = 0
		replay[3] = position_change

@numba.njit
def train(position, steps, save_period, average_reward, force_parameters,
		  value_parameters, reward_learning_rate, replay_buffer):
	replay_index = 0
	rewards = np.zeros(int(steps / save_period))
	features = fourier_basis_single(position)
	previous_position = position
	for step in range(steps):
		batch_index = numpy.random.randint(0, replays_saved - 1, batch_size)
		batch = replay_buffer[batch_index]
		previous_features = fourier_basis(batch[:,0])
		next_features = fourier_basis(batch[:,1])
		previous_values = previous_features @ value_parameters
		next_values = next_features @ value_parameters
		#reward = regularized_rewards(batch[:,0], previous_features, batch[:,3], 
		#							  force_parameters)
		forces = previous_features @ force_parameters
		reward = instant_regularized_reward(batch[:,0], forces)
		td_errors = next_values + reward - average_reward - previous_values
		parameterized_differences = batch[:,3] - time_step * forces
		differences = batch[:,3] - time_step * batch[:,2]
		ratio_exponents = -parameterized_differences**2/divisor + differences**2/divisor
		importance_ratio = np.exp(ratio_exponents)
		td_errors = td_errors * importance_ratio
		value_parameters += value_learning_rate * td_errors @ previous_features
		update = 0.5 * (td_errors * parameterized_differences) @ previous_features
		force_parameters += force_learning_rate * update
		average_reward += reward_learning_rate * np.sum(td_errors)
		reward_learning_rate += linear_rate_change

		unit_noise = numpy.random.randn()
		noise = variance * unit_noise
		parameterized_force = force_parameters @ features
		position_change = time_step * parameterized_force + noise
		position += position_change
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_basis_single(position)

		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
		if step % replacement_rate == 0:
			replay_buffer[replay_index][0] = previous_position
			replay_buffer[replay_index][1] = position
			replay_buffer[replay_index][2] = parameterized_force
			replay_buffer[replay_index][3] = position_change
			replay_index = (replay_index + 1) % replays_saved
		previous_position = position
	return rewards

#sample_trajectory = trajectory(0, 10)
#initial_time = time.time()
#sample_trajectory = trajectory(0, 100000)
#print(time.time() - initial_time)
#plt.plot(sample_trajectory)
#plt.show()

fill_buffer(0, replay_buffer, 10)
#batch = gen.choice(replay_buffer, batch_size)
#print(batch)
#print(batch[:,0])
#features = fourier_basis(batch[:,0])
#print(features)
#print(features @ value_parameters)

rewards = train(0, 100, 10, average_reward, force_parameters, value_parameters, 
				reward_learning_rate, replay_buffer)
initial_time = time.time()
rewards = train(0, training_steps, save_rate, average_reward, force_parameters, 
				value_parameters, reward_learning_rate, replay_buffer)
print(time.time() - initial_time)
print(rewards[-1])
print(force_parameters)

plt.plot(rewards)
plt.show()