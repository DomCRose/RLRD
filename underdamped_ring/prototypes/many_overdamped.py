import math
import copy
import time
import numba
import numpy as np
import numpy.random
from matplotlib import pyplot as plt
import matplotlib.animation as animation

particle_number = 40
interaction_range = 1
density = 0.5
boundary = (particle_number * interaction_range) / density
print(boundary)
time_step = 0.0001
divisor = 4 * time_step
variance = (1*time_step)**0.5
wca_scaling = 1
WCA_A = 48 * interaction_range**12 * wca_scaling
WCA_B = 24 * interaction_range**6 * wca_scaling
Activity_A = 48 * 11 * interaction_range**12 * wca_scaling
Activity_B = 24 * 5 * interaction_range**6 * wca_scaling
cut_off = interaction_range * 2**(0.166666)
left_cut_off = -cut_off
periodic_cut_off = 0.5 * boundary
average_reward = 0
reward_learning_rate = 10**(-5)
final_learning_rate = 10**(-5)
training_steps = 20000000
save_rate = 100
linear_rate_change = (final_learning_rate - reward_learning_rate)/training_steps
laguerre_basis_dimension = 10
distance_scaling = 2/boundary
distance_shift = 1
force_learning_rate = 10**(-8)
value_learning_rate = 10**(-10)
bias = -2.0

force_parameters = np.zeros(laguerre_basis_dimension)
force_current_update = np.zeros(laguerre_basis_dimension)
value_parameters = np.zeros(laguerre_basis_dimension)
value_current_update = np.zeros(laguerre_basis_dimension)

current_force = 0
current_value = 0
previous_value = 0


@numba.njit
def wca_1d_single(left_position, position, right_position):
	force = 0
	left_distance = left_position - position
	if left_distance > 0:
		left_distance -= boundary
	if left_distance > left_cut_off:
		force -= WCA_A * left_distance**(-11)
		force += WCA_B * left_distance**(-5)
	right_distance = right_position - position
	if right_distance < 0:
		right_distance += boundary
	if right_distance < cut_off:
		force -= WCA_A * right_distance**(-11)
		force += WCA_B * right_distance**(-5)
	return force

@numba.njit
def force(positions):
	forces = np.zeros(particle_number)
	for particle_index in range(particle_number - 1):
		forces[particle_index] = wca_1d_single(positions[particle_index - 1],
											   positions[particle_index],
											   positions[particle_index + 1])
	forces[particle_number - 1] = wca_1d_single(positions[particle_number - 2],
												positions[particle_number - 1],
												positions[0])
	return forces

@numba.njit
def laguerre(x):
	laguerre_values = np.zeros(laguerre_basis_dimension)
	laguerre_values[0] = 1
	laguerre_values[1] = 1 - x
	for i in range(2, laguerre_basis_dimension):
		laguerre_values[i] = (2*(i-1) + 1 - x)*laguerre_values[i-1]
		laguerre_values[i] -= (i-1)*laguerre_values[i-2]
		laguerre_values[i] /= i
	return laguerre_values

#distances = np.arange(101)*boundary/(200)
#transformed_dist = distance_shift - distance_scaling * distances
#plt.plot(distances)
#plt.plot(transformed_dist)
#plt.show()
#
#laguerre_vals = np.zeros((101, laguerre_basis_dimension))
#for i in range(101):
#	laguerre_vals[i] = laguerre(transformed_dist[i])
#plt.plot(laguerre_vals)
#plt.show()
#
#for i in range(101):
#	laguerre_vals[i] *= math.exp(-transformed_dist[i]/2)
#plt.plot(laguerre_vals)
#plt.show()
#
#laguerre_vals -= 1
#plt.plot(laguerre_vals)
#plt.show()

@numba.njit
def laguerre_basis(positions):
	features = np.zeros((particle_number, laguerre_basis_dimension))
	for particle_index in range(particle_number):
		for second_particle in range(particle_number):
			if particle_index != second_particle:
				distance = positions[particle_index] - positions[second_particle]
				if distance > 0.5*boundary:
					distance -= boundary
				if distance < -0.5*boundary:
					distance += boundary
				transformed_dist = distance_shift - distance_scaling * np.abs(distance)
				laguerre_vals = laguerre(transformed_dist)
				laguerre_vals *= math.exp(-transformed_dist*0.5)
				laguerre_vals -= 1
				laguerre_vals *= np.sign(distance)
				features[particle_index] += laguerre_vals
	return features

@numba.njit
def trajectory(positions, length, save_rate = 1):
	saves = math.floor(length / save_rate)
	trajectory = np.zeros((saves + 1, particle_number))
	trajectory[0] = positions
	save_index = 0
	for i in range(1, length + 1):
		noise = variance * numpy.random.randn(particle_number)
		positions += time_step * force(positions) + noise
		positions -= boundary*np.floor(positions / (boundary))
		if i % save_rate == 0:
			save_index += 1
			trajectory[save_index] = positions
	return trajectory

@numba.njit
def activity_1d_single(left_position, position, right_position):
	activity = 0
	left_distance = left_position - position
	if left_distance > 0:
		left_distance -= boundary
	if left_distance > left_cut_off:
		activity += Activity_A * left_distance**(-10)
		activity -= Activity_B * left_distance**(-4)
	right_distance = right_position - position
	if right_distance < 0:
		right_distance += boundary
	if right_distance < cut_off:
		activity += Activity_A * right_distance**(-10)
		activity -= Activity_B * right_distance**(-4)
	return activity

@numba.njit
def activity_derivative(positions):
	activity = 0
	for particle_index in range(particle_number - 1):
		activity += activity_1d_single(positions[particle_index - 1],
									   positions[particle_index],
									   positions[particle_index + 1])
	activity += activity_1d_single(positions[particle_number - 2],
								   positions[particle_number - 1],
								   positions[0])
	return activity


@numba.njit
def regularized_reward(positions, wca, position_change, noise):
	reward = noise**2/divisor	
	reward -= (position_change - time_step * wca)**2/divisor
	reward = np.sum(reward)
	activity = np.sum(wca ** 2)
	deriv = activity_derivative(positions)
	activity += deriv
	activity *= 0.5
	reward -= bias * time_step * activity
	return reward, activity, deriv

@numba.njit
def train(positions, steps, save_period, average_reward, force_parameters,
		  value_parameters, reward_learning_rate):
	saves = int(steps / save_period)
	average_rewards = np.zeros(saves)
	force_params = np.zeros((saves, laguerre_basis_dimension))
	value_params = np.zeros((saves, laguerre_basis_dimension))
	trajectory = np.zeros((saves, particle_number))
	wca_vs_time = np.zeros(saves)
	noises = np.zeros(saves)
	rewards = np.zeros(saves)
	activities = np.zeros(saves)
	derivs = np.zeros(saves)
	features = laguerre_basis(positions)
	current_value = np.sum(np.abs(features), axis = 0) @ value_parameters
	next_force = features @ force_parameters
	for step in range(steps):
		previous_value = current_value
		previous_features = np.copy(features)
		unit_noise = numpy.random.randn(particle_number)
		noise = variance * unit_noise
		parameterized_force = next_force
		wca = force(positions)
		full_force = parameterized_force + wca
		position_change = time_step * full_force + noise
		reward, activity, deriv = regularized_reward(positions, wca, 
									position_change, noise)
		positions += position_change
		positions -= boundary*np.floor(positions / boundary)
		next_force = features @ force_parameters
		features = laguerre_basis(positions)
		current_value = np.sum(np.abs(features), axis = 0) @ value_parameters
		td_error = current_value + reward - average_reward - previous_value
		force_parameters += (force_learning_rate * td_error
							 * unit_noise @ previous_features)
		value_parameters += (value_learning_rate * td_error 
							 * np.sum(np.abs(previous_features), axis = 0))
		average_reward += reward_learning_rate * td_error
		reward_learning_rate += linear_rate_change
		if step % save_period == 0:
			average_rewards[int(step/save_period)] = average_reward / time_step
			force_params[int(step/save_period)] = force_parameters
			value_params[int(step/save_period)] = value_parameters
			trajectory[int(step/save_period)] = positions
			wca_vs_time[int(step/save_period)] = np.sum(wca)
			noises[int(step/save_period)] = np.sum(noise)
			rewards[int(step/save_period)] = reward
			activities[int(step/save_period)] = activity
			derivs[int(step/save_period)] = deriv
	return average_rewards, force_params, value_params, trajectory, wca_vs_time, noises, rewards, activities, derivs

def periodic_animation(data, particles, boundary, interaction_range, 
					   fps = 60, circle_points = 30):
	ring_radius = boundary / (2*math.pi)
	plt.rc('font', size = 20)
	plt.rc('text', usetex = True)
	fig = plt.figure(figsize = (12, 12))
	ax = plt.axes(xlim = (- 1.2 * ring_radius, 1.2 * ring_radius), 
				  ylim = (- 1.2 * ring_radius, 1.2 * ring_radius))
	plt.xlabel(r'$x$')
	plt.ylabel(r'$y$')

	angles = np.arange(circle_points + 1) * (2 * np.pi / circle_points)
	base_circle_x = 0.5 * interaction_range * np.cos(angles)[:, np.newaxis]
	base_circle_y = 0.5 * interaction_range * np.sin(angles)[:, np.newaxis]
	circle_data_x = np.array(np.zeros((circle_points, particles)))
	circle_data_y = np.array(np.zeros((circle_points, particles)))
	circles = plt.plot(circle_data_x, circle_data_y, lw = 2, c = 'k')

	def animate(frame_data):
		particle_angles = (2*math.pi*frame_data)/boundary
		particle_y = ring_radius * np.sin(particle_angles)
		particle_x = - ring_radius * np.cos(particle_angles)
		circle_data_x = base_circle_x + particle_x
		circle_data_y = base_circle_y + particle_y
		for i in range(particles):
			circles[i].set_data(circle_data_x[:,i], circle_data_y[:,i])
		return circles

	anim = animation.FuncAnimation(fig, animate, frames = data, 
								   interval = int(1000 / fps), blit = True)
	return anim



positions = np.array([1 + 2.0*i for i in range(particle_number)])
print(positions)
#print(positions)
#
#sample_trajectory = trajectory(positions, 10, 1)
#initial_time = time.time()
#sample_trajectory = trajectory(positions, 1000000, 1000)
#print(time.time() - initial_time)
#plt.plot(sample_trajectory)
#plt.show()
#
#anim = periodic_animation(sample_trajectory, particle_number, boundary, interaction_range)
#plt.show()

output = train(positions, 100, 10, average_reward, force_parameters, value_parameters, 
				reward_learning_rate)
initial_time = time.time()
av_rewards, force_params, value_params, sample_traj, wca, noise, rewards, activity, deriv = train(positions, training_steps, save_rate, average_reward, force_parameters, 
				value_parameters, reward_learning_rate)
print(time.time() - initial_time)

plt.figure()
plt.subplot2grid((3,4), (0,0), colspan = 4)
plt.plot(sample_traj)
plt.subplot2grid((3,4), (1,0))
plt.plot(av_rewards)
plt.subplot2grid((3,4), (1,1))
plt.plot(force_params)
plt.subplot2grid((3,4), (1,2))
plt.plot(value_params)
plt.subplot2grid((3,4), (1,3))
plt.plot(wca)
plt.subplot2grid((3,4), (2,0))
plt.plot(noise)
plt.subplot2grid((3,4), (2,1))
plt.plot(rewards)
plt.subplot2grid((3,4), (2,2))
plt.plot(activity)
plt.subplot2grid((3,4), (2,3))
plt.plot(deriv)
plt.show()