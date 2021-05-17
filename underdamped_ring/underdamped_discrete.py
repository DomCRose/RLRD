import math
import copy
import time
import numba
import numpy as np
import numpy.random
from matplotlib import pyplot as plt

magnitude = 2
driving = 1
bias = -0.3
time_step = 0.001
divisor = 4 * time_step
variance = (2*time_step)**0.5
average_reward = 0
average_kl = 0
average_velocity = 0
reward_learning_rate = 10**(-6)
final_learning_rate = 10**(-6)
#training_steps = 10
training_steps = 1000000000
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
fourier_dimension = 5
spacing = 0.01
velocity_lower_limit = -8
velocity_upper_limit = 6
discrete_velocity_dimension = math.floor((velocity_upper_limit - velocity_lower_limit) 
										 / spacing)
print(discrete_velocity_dimension)
#discrete_velocity_dimension = 500
#centre = 1-bias
#spread = 4.5
#velocity_lower_limit = centre - spread
#velocity_upper_limit = centre + spread
#velocity_lower_limit = -1
#velocity_upper_limit = 3
#spacing = (velocity_upper_limit - velocity_lower_limit) / discrete_velocity_dimension
force_learning_rate = 10**(-1)
value_learning_rate = 10**(-1)

basis_index = np.arange(fourier_dimension) + 1
# Coefficients range from 0 to discrete_velocity_dimension + 1 in the velocity (first)
# index, with these limits representing v from the lower bound to -inf and the upper 
# bound to +inf respectively.
force_parameters = np.zeros((discrete_velocity_dimension + 2, fourier_dimension * 2 + 1))
for i in range(discrete_velocity_dimension + 2):
	force_parameters[i,0] = 1 - (i*spacing + velocity_lower_limit)
	force_parameters[i,1] = 2
value_parameters = np.zeros((discrete_velocity_dimension + 2, fourier_dimension * 2 + 1))
previous_features = np.zeros((discrete_velocity_dimension + 2, fourier_dimension * 2 + 1))

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
def parameterized_trajectory(position, velocity, length, force_parameters):
	features = fourier_basis(position)
	vel_index = velocity_index(velocity)
	trajectory = np.zeros((length + 1, 2))
	trajectory[0,0] = position
	trajectory[0,1] = velocity
	for i in range(1, length + 1):
		noise = variance * numpy.random.randn()
		parameterized_force = force_parameters[vel_index] @ features
		velocity += time_step * parameterized_force + noise 
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_basis(position)
		vel_index = velocity_index(velocity)
		trajectory[i,0] = position
		trajectory[i,1] = velocity
	return trajectory

@numba.njit
def regularized_reward(position, velocity, velocity_change, noise, bias):
	reward = noise**2/divisor	
	reward -= (velocity_change - time_step * (force(position)-velocity))**2/divisor
	kl_reg = reward
	#print("KL")
	#print(reward)
	#print("observable")
	#print(-bias * velocity * time_step)
	reward -= bias * velocity * time_step
	#print(reward)
	return reward, kl_reg

@numba.njit
def train(position, velocity, steps, save_period, average_reward, force_parameters,
		  value_parameters, reward_learning_rate, bias, average_kl, average_velocity):
	rewards = np.zeros(int(steps / save_period))
	kl_divergences = np.zeros(int(steps / save_period))
	velocities = np.zeros(int(steps / save_period))
	features = fourier_basis(position)
	vel_index = velocity_index(velocity)
	current_value = value_parameters[vel_index] @ features
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
		reward, kl_reg = regularized_reward(position, velocity, velocity_change, 
											noise, bias)
		velocity += velocity_change
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_basis(position)
		vel_index = velocity_index(velocity)
		current_value = value_parameters[vel_index] @ features
		#print(current_value)
		td_error = current_value + reward - average_reward - previous_value
		#print(td_error)
		force_parameters[previous_vel_index] += (force_learning_rate 
			* td_error * unit_noise * previous_features)
		value_parameters[previous_vel_index] += (value_learning_rate 
			* td_error * previous_features)
		average_reward += reward_learning_rate * td_error
		average_kl += reward_learning_rate * (kl_reg - average_kl)
		average_velocity += reward_learning_rate * (velocity - average_velocity)
		reward_learning_rate += linear_rate_change
		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
			kl_divergences[int(step/save_period)] = average_kl / time_step
			velocities[int(step/save_period)] = average_velocity
	return rewards, kl_divergences, velocities, force_parameters, value_parameters, average_reward, average_velocity, average_kl

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

bias_number = 0
initial_bias = 1.3
bias_range = 2
#bias_step = (initial_bias + bias_range)/bias_number
bias_step = 1
biases = [round(initial_bias + bias_step*i, 3) for i in range(bias_number + 1)]

for bias in biases:
	initial_time = time.time()
	rewards, kl_divergences, velocities, force_parameters, value_parameters, average_reward, average_velocity, average_kl = train(
		0, -1, training_steps, int(training_steps/10000), average_reward, 
		force_parameters, 
		value_parameters, reward_learning_rate, bias, average_kl, average_velocity)
	print(bias)
	print(time.time() - initial_time)
	#np.save("underdamped_bias%s_vl%s_vu%s_vs%s_fb%s_ts%s_force_parameters"%(
	#	bias, velocity_lower_limit, velocity_upper_limit, spacing, fourier_dimension,
	#	time_step), force_parameters)
	#np.save("underdamped_bias%s_vl%s_vu%s_vs%s_fb%s_ts%s_value_parameters"%(
	#	bias, velocity_lower_limit, velocity_upper_limit, spacing, fourier_dimension,
	#	time_step), value_parameters)
	#np.save("underdamped_bias%s_vl%s_vu%s_vs%s_fb%s_ts%s_ls%s_scgf_estimate"%(
	#	bias, velocity_lower_limit, velocity_upper_limit, spacing, fourier_dimension,
	#	time_step, training_steps), rewards)
	#np.save("underdamped_bias%s_vl%s_vu%s_vs%s_fb%s_ts%s_ls%s_kl_estimate"%(
	#	bias, velocity_lower_limit, velocity_upper_limit, spacing, fourier_dimension,
	#	time_step, training_steps), kl_divergences)
	#np.save("underdamped_bias%s_vl%s_vu%s_vs%s_fb%s_ts%s_ls%s_velocity_estimate"%(
	#	bias, velocity_lower_limit, velocity_upper_limit, spacing, fourier_dimension,
	#	time_step, training_steps), velocities)


plt.figure(figsize = (13, 4))
plt.subplot(131)
plt.plot(rewards)
plt.subplot(132)
plt.plot(kl_divergences)
plt.subplot(133)
plt.plot(velocities)
plt.show()

plt.pcolor(force_grid(force_parameters, 300, -8, 6).T)
plt.colorbar()
plt.show()
