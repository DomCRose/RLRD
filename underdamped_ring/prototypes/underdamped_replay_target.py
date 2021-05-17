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
reward_learning_rate = 10**(-4) / batch_size
final_learning_rate = 10**(-4) / batch_size
training_steps = 300000
save_rate = 10
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
fourier_dimension = 5
fourier_modes = fourier_dimension * 2 + 1
polynomial_dimension = 5
feature_dimension = fourier_modes * polynomial_dimension
force_learning_rate = 10**(-2) / batch_size
value_learning_rate = 10**(-2) / batch_size
bias = -0.3
replays_saved = 100000
replay_buffer = np.zeros((replays_saved,6))
replacement_rate = 10
target_change_rate = 0.1

basis_index = np.arange(fourier_dimension) + 1
polynomial_powers = np.arange(polynomial_dimension)
padding_array = np.array([1 for i in range(batch_size)])[:,np.newaxis]
force_parameters = np.zeros(feature_dimension)
force_parameters[0] = 1
force_parameters[1] = -1
force_parameters[polynomial_dimension] = 2
value_parameters = np.zeros(feature_dimension)
target_value_parameters = np.zeros(feature_dimension)
features = np.zeros(feature_dimension)
features[0] = 1
previous_features = np.zeros(feature_dimension)

current_force = 0
current_value = 0
previous_value = 0


@numba.njit
def force(positions):
	return magnitude * np.sin(positions) + driving

@numba.njit
def fourier_polynomial_basis(positions, velocities):
	features = np.zeros((batch_size, feature_dimension))
	for sample in range(batch_size):
		features[sample] = fourier_polynomial_basis_single(positions[sample], 
														   velocities[sample])
	return features

@numba.njit
def fourier_polynomial_basis_single(position, velocity):
	trig_arguments = position * basis_index
	sin_basis = np.sin(trig_arguments)
	cos_basis = np.cos(trig_arguments)
	fourier_basis = np.concatenate((np.array([1]), sin_basis, cos_basis))
	polynomial_basis = velocity**polynomial_powers
	polynomial_basis = np.minimum(polynomial_basis, 5)
	polynomial_basis = np.maximum(polynomial_basis, -5)
	return numpy.kron(fourier_basis, polynomial_basis)

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
def regularized_rewards(positions, velocities, features, velocity_changes, 
						force_parameters):
	#print("Kl div")
	reward = (velocity_changes - time_step * features @ force_parameters)**2/divisor
	#print(reward)
	reward -= (velocity_changes - time_step * (force(positions) - velocities))**2/divisor
	#print(reward)
	#print()
	reward -= bias * velocities * time_step
	return reward

@numba.njit
def fill_buffer(position, velocity, replay_buffer, spacing):
	for replay in replay_buffer:
		for steps in range(spacing):
			unit_noise = numpy.random.randn()
			velocity_change = (time_step * (force(position) - velocity) + variance 
							   * unit_noise)
			velocity += velocity_change
			position += velocity * time_step
			position -= 2*math.pi*math.floor(position / (2*math.pi))
		unit_noise = numpy.random.randn()
		frc = force(position) - velocity
		velocity_change = time_step * frc + variance * unit_noise 
		replay[0] = position
		replay[4] = velocity
		velocity += velocity_change
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		replay[1] = position
		replay[5] = velocity
		replay[2] = frc
		replay[3] = velocity_change

@numba.njit
def train(position, velocity, steps, save_period, average_reward, force_parameters,
		  value_parameters, reward_learning_rate, replay_buffer, target_value_parameters):
	replay_index = 0
	rewards = np.zeros(int(steps / save_period))
	features = fourier_polynomial_basis_single(position, velocity)
	previous_position = position
	previous_velocity = velocity
	for step in range(steps):
		#print(average_reward)
		#print("Forces")
		#print(force_parameters)
		batch_index = numpy.random.randint(0, replays_saved - 1, batch_size)
		batch = replay_buffer[batch_index]
		previous_features = fourier_polynomial_basis(batch[:,0], batch[:,4])
		next_features = fourier_polynomial_basis(batch[:,1], batch[:,5])
		previous_values = previous_features @ target_value_parameters
		next_values = next_features @ target_value_parameters
		reward = regularized_rewards(batch[:,0], batch[:,4], previous_features, 
									 batch[:,3], force_parameters)
		td_errors = next_values + reward - average_reward - previous_values
		#td_errors = np.minimum(td_errors, 5)
		#td_errors = np.maximum(td_errors, -5)
		forces = previous_features @ force_parameters
		parameterized_differences = batch[:,3] - time_step * forces
		differences = batch[:,3] - time_step * batch[:,2]
		ratio_exponents = -parameterized_differences**2/divisor + differences**2/divisor
		importance_ratio = np.exp(ratio_exponents)
		#importance_ratio = np.minimum(importance_ratio, 5)
		#print("Start")
		#for i in range(batch_size):
			#print(ratio_exponents[i])
			#print(importance_ratio[i])
			#print(td_errors[i])
		#print()
		td_errors = td_errors * importance_ratio
		#print(td_errors)
		value_parameters += value_learning_rate * td_errors @ previous_features
		target_value_parameters += target_change_rate * (
			value_parameters - target_value_parameters)
		update = 0.5 * (td_errors * parameterized_differences) @ previous_features
		force_parameters += force_learning_rate * update
		#print(np.sum(td_errors))
		average_reward += reward_learning_rate * np.sum(td_errors)
		reward_learning_rate += linear_rate_change

		unit_noise = numpy.random.randn()
		noise = variance * unit_noise
		parameterized_force = force_parameters @ features
		velocity_change = time_step * parameterized_force + noise
		velocity += velocity_change
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		features = fourier_polynomial_basis_single(position, velocity)

		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
		if step % replacement_rate == 0:
			replay_buffer[replay_index][0] = previous_position
			replay_buffer[replay_index][4] = previous_velocity
			replay_buffer[replay_index][1] = position
			replay_buffer[replay_index][5] = velocity
			replay_buffer[replay_index][2] = parameterized_force
			replay_buffer[replay_index][3] = velocity_change
			replay_index = (replay_index + 1) % replays_saved
		previous_position = position
		previous_velocity = velocity
	return rewards

#sample_trajectory = trajectory(0, 10)
#initial_time = time.time()
#sample_trajectory = trajectory(0, 100000)
#print(time.time() - initial_time)
#plt.plot(sample_trajectory)
#plt.show()

fill_buffer(0, 0, replay_buffer, 100)
#batch = gen.choice(replay_buffer, batch_size)
#print(batch)
#print(batch[:,0])
#features = fourier_basis(batch[:,0])
#print(features)
#print(features @ value_parameters)

rewards = train(0, 1, 2, 10, average_reward, force_parameters, value_parameters, 
				reward_learning_rate, replay_buffer, target_value_parameters)
initial_time = time.time()
rewards = train(0, 1, training_steps, save_rate, average_reward, force_parameters, 
				value_parameters, reward_learning_rate, replay_buffer, 
				target_value_parameters)
print(time.time() - initial_time)
print(rewards[-1])

plt.plot(rewards)
plt.show()