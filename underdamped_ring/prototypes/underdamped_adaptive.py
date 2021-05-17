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
reward_learning_rate = 10**(-6)
final_learning_rate = 10**(-6)
training_steps = 10000000
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
fourier_dimension = 3
polynomial_dimension = 2
force_learning_rate = 10**(-6)
value_learning_rate = 10**(-5.5)
forgetting_rate = 0.5
bias = -0.5

basis_index = np.arange(fourier_dimension) + 1
polynomial_powers = np.arange(polynomial_dimension)
force_parameters = np.zeros((fourier_dimension * 2 + 1, polynomial_dimension))
force_rms = np.zeros((fourier_dimension * 2 + 1, polynomial_dimension))
force_parameters[0][0] = 1
force_parameters[0][1] = -1
force_parameters[1][0] = 2
value_parameters = np.zeros((fourier_dimension * 2 + 1, polynomial_dimension))
value_rms = np.zeros((fourier_dimension * 2 + 1, polynomial_dimension))
previous_features = np.zeros((fourier_dimension * 2 + 1, polynomial_dimension))

current_force = 0
current_value = 0
previous_value = 0


@numba.njit
def force(position):
	return magnitude * math.sin(position) + driving

@numba.njit
def fourier_polynomial_basis(position, velocity):
	trig_arguments = position * basis_index
	sin_basis = np.sin(trig_arguments)
	cos_basis = np.cos(trig_arguments)
	fourier_basis = np.concatenate((np.array([1]), sin_basis, cos_basis))
	polynomial_basis = velocity**polynomial_powers
	#polynomial_basis = np.minimum(polynomial_basis, 5)
	#polynomial_basis = np.maximum(polynomial_basis, -5)
	return numpy.outer(fourier_basis, polynomial_basis)

@numba.njit
def trajectory(position, velocity, length):
	trajectory = np.zeros((length + 1, 2))
	trajectory[0,0] = position
	trajectory[0,1] = velocity
	for i in range(1, length + 1):
		noise = variance * numpy.random.randn()
		velocity += time_step * (force(position) - velocity) + noise 
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		trajectory[i,0] = position
		trajectory[i,1] = velocity
	return trajectory

@numba.njit
def regularized_reward(position, velocity, velocity_change, noise):
	reward = noise**2/divisor	
	reward -= (velocity_change - time_step * (force(position)-velocity))**2/divisor
	reward -= bias * velocity * time_step
	return reward

@numba.njit
def pretrain(position, velocity, steps, save_period, average_reward, force_parameters,
			  value_parameters, reward_learning_rate):
	rewards = np.zeros(int(steps / save_period))
	value_params = np.zeros((int(steps / save_period), 
							 (fourier_dimension * 2 + 1) * polynomial_dimension))
	features = fourier_polynomial_basis(position, velocity)
	current_value = np.sum(value_parameters * features)
	for step in range(steps):
		previous_value = current_value
		previous_features = np.copy(features)
		unit_noise = numpy.random.randn()
		noise = variance * unit_noise
		parameterized_force = np.sum(force_parameters * features)
		velocity_change = time_step * parameterized_force + noise
		reward = regularized_reward(position, velocity, velocity_change, noise)
		velocity += velocity_change
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_polynomial_basis(position, velocity)
		current_value = np.sum(value_parameters * features)
		td_error = current_value + reward - average_reward - previous_value
		value_parameters += value_learning_rate * td_error * previous_features
		average_reward += reward_learning_rate * td_error
		reward_learning_rate += linear_rate_change
		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
			value_params[int(step/save_period)] += value_parameters.flatten()
	return rewards, value_params

@numba.njit
def train(position, velocity, steps, save_period, average_reward, force_parameters,
		  value_parameters, reward_learning_rate, force_rms, value_rms):
	rewards = np.zeros(int(steps / save_period))
	force_params = np.zeros((int(steps / save_period), 
							 (fourier_dimension * 2 + 1) * polynomial_dimension))
	value_params = np.zeros((int(steps / save_period), 
							 (fourier_dimension * 2 + 1) * polynomial_dimension))
	features = fourier_polynomial_basis(position, velocity)
	current_value = np.sum(value_parameters * features)
	for step in range(steps):
		previous_value = current_value
		previous_features = np.copy(features)
		unit_noise = numpy.random.randn()
		noise = variance * unit_noise
		parameterized_force = np.sum(force_parameters * features)
		velocity_change = time_step * parameterized_force + noise
		reward = regularized_reward(position, velocity, velocity_change, noise)
		velocity += velocity_change
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_polynomial_basis(position, velocity)
		current_value = np.sum(value_parameters * features)
		td_error = current_value + reward - average_reward - previous_value
		force_gradient = td_error * unit_noise * previous_features
		force_rms = (forgetting_rate * force_rms 
					 + (1-forgetting_rate) * force_gradient**2)
		force_parameters += (force_learning_rate 
							 * force_gradient) / (np.sqrt(force_rms) + 10**(-8))
		value_gradient = td_error * previous_features
		value_rms = (forgetting_rate * value_rms 
					 + (1-forgetting_rate) * value_gradient**2)
		value_parameters += (value_learning_rate 
							 * value_gradient) / (np.sqrt(value_rms) + 10**(-8))
		average_reward += reward_learning_rate * td_error
		reward_learning_rate += linear_rate_change
		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
			force_params[int(step/save_period)] += force_parameters.flatten()
			value_params[int(step/save_period)] += value_parameters.flatten()
	return rewards, force_params, value_params

#sample_trajectory = trajectory(0, 0, 1)
#initial_time = time.time()
#sample_trajectory = trajectory(0, 0, 10000)
#print(time.time() - initial_time)
#plt.plot(sample_trajectory)
#plt.show()

#pretraining_steps = training_steps
#rewards, value_params = pretrain(0, 1, pretraining_steps, 10, average_reward, #force_parameters, value_parameters, 
#				reward_learning_rate)
#plt.figure(figsize = (9*1.0, 4))
#plt.subplot(121)
#plt.plot(rewards)
#plt.subplot(122)
#plt.plot(value_params)
#plt.show()	
#print(average_reward)
rewards, force_params, value_params = train(0, 1, 10, 10, average_reward, force_parameters, value_parameters, 
				reward_learning_rate, force_rms, value_rms)
initial_time = time.time()
rewards, force_params, value_params = train(0, 1, training_steps, 1000, average_reward, force_parameters, 
				value_parameters, reward_learning_rate, force_rms, value_rms)
print(time.time() - initial_time)
print(force_parameters)
print(rewards[-1])

plt.figure(figsize = (9*1.5, 4))
plt.subplot(131)
plt.plot(rewards)
plt.subplot(132)
plt.plot(force_params)
plt.subplot(133)
plt.plot(value_params)
plt.show()