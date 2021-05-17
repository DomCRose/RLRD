import math
import copy
import time
import numba
import numpy as np
import numpy.random
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

magnitude = 2
driving = 1
bias = -0.3
time_step = 0.001
divisor = 4 * time_step
variance = (2*time_step)**0.5
average_reward = 0
reward_learning_rate = 10**(-6)
final_learning_rate = 10**(-7)
training_steps = 10000000
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
ss_learning_rate = 10**(-4)
fourier_dimension = 5
discrete_velocity_dimension = 500
centre = 1-bias
spread = 4.5
velocity_lower_limit = centre - spread
velocity_upper_limit = centre + spread
#velocity_lower_limit = -1
#velocity_upper_limit = 3
spacing = (velocity_upper_limit - velocity_lower_limit) / discrete_velocity_dimension

force_learning_rate = 10**(-1)
value_learning_rate = 10**(-1)

basis_index = np.arange(fourier_dimension) + 1
# Coefficients range from 0 to discrete_velocity_dimension + 1 in the velocity (first)
# index, with these limits representing v from the lower bound to -inf and the upper 
# bound to +inf respectively.
force_parameters = np.zeros((discrete_velocity_dimension + 2, fourier_dimension * 2 + 1))
value_parameters = np.zeros((discrete_velocity_dimension + 2, fourier_dimension * 2 + 1))
previous_features = np.zeros((discrete_velocity_dimension + 2, fourier_dimension * 2 + 1))
discrete_position_dimension = 100
steady_state_bin_width = 2*math.pi / discrete_position_dimension
steady_state_parameters = np.zeros((discrete_velocity_dimension + 2, 
									discrete_position_dimension))

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
	fourier_basis = np.concatenate((np.array([1]), sin_basis, cos_basis))
	return fourier_basis

@numba.njit
def velocity_index(velocity):
	uncapped_index = math.ceil((velocity - velocity_lower_limit)/spacing)
	if uncapped_index > discrete_velocity_dimension:
		index = discrete_velocity_dimension + 1
	elif uncapped_index < 0:
		index = 0
	else:
		index = uncapped_index
	return index

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
	#print("KL")
	#print(reward)
	#print("observable")
	#print(-bias * velocity * time_step)
	reward -= bias * velocity * time_step
	#print(reward)
	return reward

@numba.njit
def train(position, velocity, steps, save_period, average_reward, force_parameters,
		  value_parameters, steady_state_parameters, reward_learning_rate):
	rewards = np.zeros(int(steps / save_period))
	features = fourier_basis(position)
	vel_index = velocity_index(velocity)
	current_value = value_parameters[vel_index] @ features
	ss_norm = np.sum(np.exp(steady_state_parameters))
	for step in range(steps):
		previous_value = current_value
		previous_features = np.copy(features)
		previous_vel_index = vel_index
		unit_noise = numpy.random.randn()
		noise = variance * unit_noise
		parameterized_force = force_parameters[vel_index] @ features
		velocity_change = time_step * parameterized_force + noise
		#print(step)
		#print(velocity_change)
		#print(velocity)
		reward = regularized_reward(position, velocity, velocity_change, noise)
		velocity += velocity_change
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_basis(position)
		vel_index = velocity_index(velocity)
		pos_index = math.floor(position / steady_state_bin_width)
		unnormalised_probability = np.exp(steady_state_parameters[vel_index][pos_index])
		probability = unnormalised_probability / ss_norm
		current_value = value_parameters[vel_index] @ features
		#print(current_value)
		td_error = current_value + reward - average_reward - previous_value
		#print(td_error)
		force_parameters[previous_vel_index] += (force_learning_rate 
			* td_error * unit_noise * previous_features)
		value_parameters[previous_vel_index] += (value_learning_rate 
			* td_error * previous_features)
		average_reward += reward_learning_rate * td_error
		reward_learning_rate += linear_rate_change
		steady_state_parameters[vel_index][pos_index] += (
			ss_learning_rate * (1 - probability))
		unnormalised_prob_new = np.exp(steady_state_parameters[vel_index][pos_index])
		ss_norm -= unnormalised_probability
		ss_norm += unnormalised_prob_new
		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
		if step % (save_period * 100) == 0:
			steady_state_parameters -= np.log(ss_norm)
			ss_norm = 1 
	return rewards

@numba.njit
def force_grid(force_parameters, position_points, velocity_min, 
			   velocity_max):
	velocity_min_index = velocity_index(velocity_min)
	velocity_max_index = velocity_index(velocity_max)
	spacing = 2*math.pi / position_points
	force = np.zeros((velocity_max_index - velocity_min_index, position_points + 1))
	for vel_index in range(velocity_min_index, velocity_max_index):
		for position_index in range(position_points + 1):
			position = spacing * position_index
			features = fourier_basis(position)
			force[vel_index - velocity_min_index][position_index] = (
				force_parameters[vel_index] @ features)
	return force

@numba.njit
def steady_state(position, velocity, steps, average_reward, force_parameters,
				 value_parameters, steady_state_parameters):
	features = fourier_basis(position)
	vel_index = velocity_index(velocity)
	ss_norm = np.sum(np.exp(steady_state_parameters))
	for step in range(steps):
		unit_noise = numpy.random.randn()
		noise = variance * unit_noise
		parameterized_force = force_parameters[vel_index] @ features
		velocity_change = time_step * parameterized_force + noise
		velocity += velocity_change
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_basis(position)
		vel_index = velocity_index(velocity)
		pos_index = math.floor(position / steady_state_bin_width)
		unnormalised_probability = np.exp(steady_state_parameters[vel_index][pos_index])
		probability = unnormalised_probability / ss_norm
		steady_state_parameters[vel_index][pos_index] += (
			ss_learning_rate * (1 - probability))
		unnormalised_prob_new = np.exp(steady_state_parameters[vel_index][pos_index])
		ss_norm -= unnormalised_probability
		ss_norm += unnormalised_prob_new
		if step % 1000 == 0:
			steady_state_parameters -= np.log(ss_norm)
			ss_norm = 1

rewards = train(0, -1, 10, 10, average_reward, force_parameters, value_parameters, 
				steady_state_parameters, 
				reward_learning_rate)
initial_time = time.time()
rewards = train(0, -1, training_steps, 100, average_reward, force_parameters, 
				value_parameters, steady_state_parameters, reward_learning_rate)
print(time.time() - initial_time)
print(force_parameters)
print(rewards[-1])

plt.plot(rewards)
plt.show()

plt.pcolormesh(force_grid(force_parameters, 300, 1, 3).T)
plt.colorbar()
plt.show()

ss_norm = np.sum(np.exp(steady_state_parameters))
probabilities = np.exp(steady_state_parameters) / ss_norm
plt.pcolormesh(probabilities, norm = LogNorm())
plt.colorbar()
plt.show()

steady_state_parameters = np.zeros((discrete_velocity_dimension + 2, 
									discrete_position_dimension))
steps = 10000000
steady_state(0, -1, steps, average_reward, force_parameters,
		  value_parameters, steady_state_parameters)
ss_norm = np.sum(np.exp(steady_state_parameters))
probabilities = np.exp(steady_state_parameters) / ss_norm
plt.pcolormesh(probabilities, norm = LogNorm())
plt.colorbar()
plt.show()