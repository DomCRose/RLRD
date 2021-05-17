import math
import copy
import time
import numba
import numpy as np
import numpy.random
from matplotlib import pyplot as plt

import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class ForceNetwork(nn.Module):
	def __init__(self):
		super(ForceNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(2, 32),
			nn.ReLU(),
			nn.Linear(32, 1)
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits


class ValueNetwork(nn.Module):
	def __init__(self):
		super(ValueNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(2, 32),
			nn.ReLU(),
			nn.Linear(32, 1)
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits

def force(positions):
	return magnitude * np.sin(positions) + driving

def trajectory(position, length):
	trajectory = np.zeros(length + 1)
	trajectory[0] = position
	for i in range(1, length + 1):
		noise = variance * numpy.random.randn()
		position += time_step * force(position) + noise
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		trajectory[i] = position
	return trajectory

def regularized_rewards(forces, batch, velocity_changes):
	reward = (velocity_changes - time_step * forces)**2/divisor
	reward -= (velocity_changes - time_step * (force(batch[:,0]) - batch[:,1]))**2/divisor
	kl_reg = -torch.clone(reward)
	reward -= bias * batch[:,1] * time_step
	#print(batch[:,1])
	#print(kl_reg)
	#print(bias * batch[:,1] * time_step)
	#print()
	return reward, kl_reg

def importance_ratio(forces, velocity_change, sampled_force):
	#print()
	#print(time_step * forces)
	#print(time_step * sampled_force)
	#print(velocity_change)
	parameterized_differences = velocity_change - time_step * forces
	differences = velocity_change - time_step * sampled_force
	#print(parameterized_differences)
	#print(differences)
	ratio_exponents = -parameterized_differences**2/divisor + differences**2/divisor
	#print(ratio_exponents)
	importance_ratio = torch.exp(ratio_exponents)
	#print(importance_ratio)
	return importance_ratio, parameterized_differences

def fill_buffer(position, velocity, spacing):
	for i  in range(replays_saved):
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
		replay_buffer_prior[i][0] = position
		replay_buffer_prior[i][1] = velocity
		velocity += velocity_change
		position += velocity * time_step
		position -= 2*math.pi*math.floor(position / (2*math.pi))
		replay_buffer_post[i][0] = position
		replay_buffer_post[i][1] = velocity
		replay_buffer_force[i] = frc
		replay_buffer_change[i] = velocity_change
	#replay_buffer_prior.to(device)
	#replay_buffer_post.to(device)
	#replay_buffer_force.to(device)
	#replay_buffer_change.to(device)

def train(position, velocity, steps, save_period,
		  force_model, value_model, target_model,
		  average_reward, average_kl, average_velocity,
		  force_optimizer, value_optimizer, reward_learning_rate, replacement_index):
	rewards = np.zeros(int(steps / save_period))
	kl_divergences = np.zeros(int(steps / save_period))
	velocities = np.zeros(int(steps / save_period))
	importances = np.zeros(int(steps / save_period))
	average_importance = 0

	for step in range(steps):
		replay_indices = torch.randint(replays_saved, (batch_size,))
		replays_prior = replay_buffer_prior[replay_indices]
		replays_post = replay_buffer_post[replay_indices]
		replays_force = replay_buffer_force[replay_indices]
		replays_change = replay_buffer_change[replay_indices]

		forces = force_model(replays_prior)[:,0]# + (force(replays_prior[:,0]) 
											   #- replays_prior[:,1])
		force_values = forces.detach().flatten()
		values = value_model(replays_prior)
		value_values = values.detach().flatten()
		with torch.no_grad():
			next_values = target_model(replays_post).flatten()
			reward, kl_reg = regularized_rewards(force_values, 
												   replays_prior, replays_change)
			importance, noise_factor = importance_ratio(force_values, 
														replays_change, replays_force)
			td_error = next_values + reward - average_reward - value_values
			scaled_td_error = td_error * importance
			#scaled_td_error = (reward - average_reward) * importance
		
		force_loss = -torch.sum(scaled_td_error * noise_factor * forces)
		value_loss = -torch.sum(scaled_td_error * values)

		force_optimizer.zero_grad()
		force_loss.backward()
		force_optimizer.step()
		value_optimizer.zero_grad()
		value_loss.backward()
		value_optimizer.step()
		average_reward += reward_learning_rate * torch.sum(reward * importance - average_reward)
		#average_reward += reward_learning_rate * torch.sum(scaled_td_error)
		average_kl += reward_learning_rate * torch.sum(kl_reg * importance - average_kl)
		average_velocity += reward_learning_rate * torch.sum(
			replays_prior[:,1] * importance - average_velocity)
		average_importance += reward_learning_rate * torch.sum(torch.abs(importance-1) 
															   - average_importance)

		if step % target_change_frequency == 0:
			target_model.load_state_dict(value_model.state_dict())

		if step % save_period == 0:
			rewards[int(step/save_period)] = average_reward / time_step
			kl_divergences[int(step/save_period)] = average_kl / time_step
			velocities[int(step/save_period)] = average_velocity
			importances[int(step/save_period)] = average_importance

		if step % 100 == 0:
			print(reward - kl_reg)
			print(kl_reg)
			print(td_error)
			print("Steps: " + str(step))
			print()

		if step % replacement_period == 0:
			for steps in range(replacement_amount):
				unit_noise = numpy.random.randn()
				with torch.no_grad():
					frc = force_model(torch.tensor([[position, velocity]]).float()).item()
						   #+ force(position) - velocity)
				velocity_change = time_step * frc + variance * unit_noise
				replay_buffer_prior[replacement_index][0] = position
				replay_buffer_prior[replacement_index][1] = velocity
				velocity += velocity_change
				position += velocity * time_step
				position -= 2*math.pi*math.floor(position / (2*math.pi))
				replay_buffer_post[replacement_index][0] = position
				replay_buffer_post[replacement_index][1] = velocity
				replay_buffer_force[replacement_index] = frc
				replay_buffer_change[replacement_index] = velocity_change
				replacement_index = (replacement_index + 1) % replays_saved
				
			#print("Replacing. Optimization step: " + str(step))
			#print(velocity)

	return rewards, kl_divergences, velocities, importances

def original_force():
	positions = np.linspace(0, 2*np.pi, num = 50)
	velocities = np.linspace(-6, 6, num = 70)
	return force(positions) - velocities[:,np.newaxis], positions, velocities

def force_grid(force_model):
	positions = np.linspace(0, 2*np.pi, num = 50)
	velocities = np.linspace(-6, 6, num = 70)
	pos, vel = np.meshgrid(positions, velocities)
	coordinates = np.resize(np.stack((pos, vel), axis = -1), (50*70, 2))
	coords_torch = torch.from_numpy(coordinates).float()
	with torch.no_grad():
		forces = force_model(coords_torch).flatten()# + (force(coords_torch[:,0]) 
													#	- coords_torch[:,1])
	return np.reshape(forces.numpy(), (70, 50))

def value_grid(value_model):
	positions = np.linspace(0, 2*np.pi, num = 50)
	velocities = np.linspace(-6, 6, num = 70)
	pos, vel = np.meshgrid(positions, velocities)
	coordinates = np.resize(np.stack((pos, vel), axis = -1), (50*70, 2))
	coords_torch = torch.from_numpy(coordinates).float()
	with torch.no_grad():
		forces = value_model(coords_torch).flatten()
	return np.reshape(forces.numpy(), (70, 50))


magnitude = 2
driving = 1
time_step = 0.001
divisor = 4 * time_step
variance = (2*time_step)**0.5
average_reward = 0
average_kl = 0
average_velocity = 0
batch_size = 32
reward_learning_rate = 10**(-6)
training_steps = 500000
save_rate = 10

force_learning_rate = 10**(-4)
value_learning_rate = 10**(-4)
bias = 1.3
replays_saved = 100000
replay_buffer_prior = torch.zeros((replays_saved,2))
replay_buffer_post = torch.zeros((replays_saved,2))
replay_buffer_change = torch.zeros((replays_saved))
replay_buffer_force = torch.zeros((replays_saved))
replacement_period = 1
replacement_amount = 100
replacement_index = 0
target_change_frequency = 100

force_model = ForceNetwork()#.to(device)
value_model = ValueNetwork()#.to(device)
target_model = ValueNetwork()#.to(device)

force_optimizer = torch.optim.Adam(force_model.parameters(), lr = force_learning_rate)
value_optimizer = torch.optim.Adam(value_model.parameters(), lr = value_learning_rate)

fill_buffer(0, 0, 1)

original_frc, pos, vel = original_force()
mesh_x, mesh_y = np.meshgrid(vel, pos)
param_val_pre_train = value_grid(value_model)
param_frc_pre_train = force_grid(force_model)
plt.figure(figsize = (13, 4))
plt.subplot(131)
plt.pcolor(mesh_x, mesh_y, original_frc.T)
plt.scatter(replay_buffer_prior[:,1].numpy(), replay_buffer_prior[:,0].numpy(), 
			marker = 'x', lw = 0.2, c = 'k')
plt.colorbar()
plt.subplot(132)
plt.pcolor(mesh_x, mesh_y, param_frc_pre_train.T)
plt.colorbar()
plt.subplot(133)
plt.pcolor(mesh_x, mesh_y, param_val_pre_train.T)
plt.colorbar()
plt.show()



initial_time = time.time()
data = train(0, 0, training_steps, 1,
			 force_model, value_model, target_model,
			 average_reward, average_kl, average_velocity,
			 force_optimizer, value_optimizer, reward_learning_rate, replacement_index)
print(time.time() - initial_time)


plt.figure(figsize = (13, 4))
plt.subplot(141)
plt.plot(data[0])
plt.subplot(142)
plt.plot(data[1])
plt.subplot(143)
plt.plot(data[2])
plt.subplot(144)
plt.plot(data[3])
plt.show()


param_frc = force_grid(force_model)
param_val = value_grid(value_model)
plt.figure(figsize = (13, 10))
plt.subplot(331)
plt.pcolor(mesh_x, mesh_y, original_frc.T, vmin = -6, vmax = 8)
plt.colorbar()
plt.subplot(332)
plt.pcolor(mesh_x, mesh_y, original_frc.T, vmin = -6, vmax = 8)
plt.colorbar()
plt.scatter(replay_buffer_prior[:,1].numpy(), replay_buffer_prior[:,0].numpy(), 
			marker = 'x', lw = 0.2, c = 'k')
plt.subplot(334)
plt.pcolor(mesh_x, mesh_y, param_frc_pre_train.T, vmin = -6, vmax = 8)
plt.colorbar()
plt.subplot(335)
plt.pcolor(mesh_x, mesh_y, param_frc.T, vmin = -6, vmax = 8)
plt.colorbar()
plt.subplot(336)
plt.pcolor(mesh_x, mesh_y, param_frc.T - original_frc.T)
plt.colorbar()
plt.subplot(337)
plt.pcolor(mesh_x, mesh_y, param_val_pre_train.T)
plt.colorbar()
plt.subplot(338)
plt.pcolor(mesh_x, mesh_y, param_val.T)#, vmin = -6, vmax = 8)
plt.colorbar()
plt.show()