import math
import copy
import time
import numba
import numpy as np
import numpy.random
import pandas as pd
from matplotlib import pyplot as plt

magnitude = 2
driving = 1
time_step = 0.001
divisor = 4 * time_step
variance = (2*time_step)**0.5
average_reward = 0
average_kl = 0
average_velocity = 0
reward_learning_rate = 10**(-6)
final_learning_rate = 10**(-6)
training_steps = 10000000
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
fourier_dimension = 5
polynomial_dimension = 2
force_learning_rate = 10**(-2)
value_pre_learning_rate = 10**(-2.5)
value_learning_rate = 10**(-3)
bias = 1.5

def parse_file(fname):
	# Read data from CSV file, assuming tabs as separators
	return pd.read_csv(fname, delimiter='\t', header = None)

#convert_csv("underdamped_v-dependent_forces.txt")
df = parse_file("underdamped_v-dependent_forces_data.csv")
df2 = df.reindex(columns = [0, 11, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5,
							   22, 17, 18, 19, 20, 21, 12, 13, 14, 15, 16])

basis_index = np.arange(fourier_dimension) + 1
polynomial_powers = np.arange(polynomial_dimension)
force_parameters = np.zeros((fourier_dimension * 2 + 1, polynomial_dimension))
force_parameters[0][0] = 1
force_parameters[0][1] = -1
force_parameters[1][0] = 2
value_parameters = np.zeros((fourier_dimension * 2 + 1, polynomial_dimension))
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
def regularized_reward(position, velocity, velocity_change, noise, bias):
	reward = noise**2/divisor	
	reward -= (velocity_change - time_step * (force(position)-velocity))**2/divisor
	kl_reg = reward
	reward -= bias * velocity * time_step
	return reward, kl_reg

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
		reward, kl_reg = regularized_reward(position, velocity, velocity_change, noise)
		velocity += velocity_change
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_polynomial_basis(position, velocity)
		current_value = np.sum(value_parameters * features)
		td_error = current_value + reward - average_reward - previous_value
		value_parameters += value_pre_learning_rate * td_error * previous_features
		average_reward += reward_learning_rate * td_error
		reward_learning_rate += linear_rate_change
		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
			value_params[int(step/save_period)] += value_parameters.flatten()
	return rewards, value_params

@numba.njit
def train(position, velocity, steps, save_period, average_reward, force_parameters,
		  value_parameters, reward_learning_rate, bias, average_kl, average_velocity):
	rewards = np.zeros(int(steps / save_period))
	kl_divergences = np.zeros(int(steps / save_period))
	velocities = np.zeros(int(steps / save_period))
	features = fourier_polynomial_basis(position, velocity)
	current_value = np.sum(value_parameters * features)
	for step in range(steps):
		previous_value = current_value
		previous_features = np.copy(features)
		unit_noise = numpy.random.randn()
		noise = variance * unit_noise
		parameterized_force = np.sum(force_parameters * features)
		velocity_change = time_step * parameterized_force + noise
		reward, kl_reg = regularized_reward(position, velocity, velocity_change, 
											noise, bias)
		#print(reward)
		velocity += velocity_change
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_polynomial_basis(position, velocity)
		current_value = np.sum(value_parameters * features)
		td_error = current_value + reward - average_reward - previous_value
		force_parameters += (force_learning_rate * td_error * unit_noise
							 * previous_features)
		value_parameters += value_learning_rate * td_error * previous_features
		#average_reward += reward_learning_rate * (reward - average_reward) 
		#average_reward += reward_learning_rate * td_error
		average_reward += reward_learning_rate * (reward - average_reward)
		average_kl += reward_learning_rate*0.1 * (kl_reg - average_kl)
		average_velocity += reward_learning_rate * (velocity - average_velocity)
		reward_learning_rate += linear_rate_change
		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
			kl_divergences[int(step/save_period)] = average_kl / time_step
			velocities[int(step/save_period)] = average_velocity
			#print(velocity)
	return rewards, kl_divergences, velocities, force_parameters, value_parameters, average_reward, average_velocity, average_kl



bias_number = 20
initial_bias = 1.5
bias_range = 2
bias_step = bias_range/bias_number
print(bias_step)
#bias_step = 1
biases = [round(initial_bias - bias_step*i, 3) for i in range(bias_number + 1)]
print(biases)

rews = []
kldivs = []
velocits = []
scgfs = []
kls = []
velos = []

print(df2.values)
print(df2.values.shape)

i = 21
for bias in biases:
	#force_parameters[:,0] = df2.values[i, 1:12].astype(np.float)
	#force_parameters[:,1] = -df2.values[i, 12:24].astype(np.float)
	#print()
	#print(round(-float(df2.values[i, 0]), 3))

	initial_time = time.time()
	rewards, kl_divergences, velocities, force_parameters, value_parameters, average_reward, average_velocity, average_kl = train(
		0, 0, training_steps, int(training_steps/10000), average_reward, 
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
	print(scgfs)
	print()
	i -= 1
	#np.save("underdamped_bias%s_fb%s_pb%s_ts%s_force_parameters"%(
	#	bias, fourier_dimension, polynomial_dimension,
	#	time_step), force_parameters)
	#np.save("underdamped_bias%s_fb%s_pb%s_ts%s_value_parameters"%(
	#	bias, fourier_dimension, polynomial_dimension,
	#	time_step), value_parameters)
	#np.save("underdamped_bias%s_fb%s_pb%s_ts%s_ls%s_scgf_estimate"%(
	#	bias, fourier_dimension, polynomial_dimension,
	#	time_step, training_steps), rewards)
	#np.save("underdamped_bias%s_fb%s_pb%s_ts%s_ls%s_kl_estimate"%(
	#	bias, fourier_dimension, polynomial_dimension,
	#	time_step, training_steps), kl_divergences)
	#np.save("underdamped_bias%s_fb%s_pb%s_ts%s_ls%s_velocity_estimate"%(
	#	bias, fourier_dimension, polynomial_dimension,
	#	time_step, training_steps), velocities)


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