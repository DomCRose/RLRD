import Random
import Plots

abstract type State end
abstract type Particle <: State end
abstract type Trajectory end
abstract type Dynamics end


mutable struct PeriodicParticle <: Particle
	position::Float64
	boundary::Float64
end

struct ParticleTrajectory <: Trajectory
	length::Int64
	positions::Array{Float64}
end
ParticleTrajectory(length) = ParticleTrajectory(length, zeros(length))
function write!(trajectory::ParticleTrajectory, state::Particle, index::Int)
	trajectory.positions[index] = state.position
	return nothing
end

function evolutions(dynamics::Dynamics, time::Number)::Int
	return ceil(time / dynamics.time_step)
end

struct BrownianMotion <: Dynamics
	time_step::Float64
	diffusion_coefficient::Float64
	variance::Float64
end
BrownianMotion(time_step, diffusion_coefficient) = BrownianMotion(
	time_step, diffusion_coefficient, sqrt(time_step*diffusion_coefficient))
function evolve!(state::PeriodicParticle, dynamics::BrownianMotion)
	noise = Random.randn()
	state.position += dynamics.variance * noise
	state.position -= state.boundary * floor(state.position / state.boundary)
	return nothing
end

struct PeriodicallyDrivenBrownian <: Dynamics
	time_step::Float64
	diffusion_coefficient::Float64
	variance::Float64
	magnitude::Float64
	driving::Float64
end
function PeriodicallyDrivenBrownian(time_step, diffusion_coefficient, magnitude, driving) 
	dynamics = PeriodicallyDrivenBrownian(time_step, diffusion_coefficient, 
										  sqrt(time_step*diffusion_coefficient), 
										  magnitude, driving)
end
function evolve!(state::PeriodicParticle, dynamics::PeriodicallyDrivenBrownian)
	noise = Random.randn()
	state.position += dynamics.variance * noise
	force = dynamics.magnitude * sin(state.position) + dynamics.driving
	state.position += dynamics.time_step * force
	state.position -= state.boundary * floor(state.position / state.boundary)
	return nothing
end

function evolve!(state::State, dynamics::Dynamics, time::Float64)
	steps = evolutions(dynamics, time)
	for step = 1:steps
		evolve!(state, dynamics)
	end
	return nothing
end

function evolve!(state::Particle, dynamics::Dynamics, 
				 time::Float64, trajectory::ParticleTrajectory)
	steps = evolutions(dynamics, time)
	write!(trajectory, state, 1)
	for step = 1:steps
		evolve!(state, dynamics)
		write!(trajectory, state, step)
	end
	return nothing
end

function periodic_driven_brownian_trajectory(state::Particle, dynamics::Dynamics, 
											 time::Float64, 
											 trajectory::ParticleTrajectory)
	steps = convert(Int64, ceil(time / dynamics.time_step))
	trajectory.positions[1] = state.position
	for step = 1:steps
		noise = Random.randn()
		state.position += dynamics.variance * noise
		force = dynamics.magnitude * sin(state.position) + dynamics.driving
		state.position += dynamics.time_step * force
		state.position -= state.boundary * floor(state.position / state.boundary)
		trajectory.positions[step] = state.position
	end
	return nothing
end

function periodic_driven_brownian_trajectory_basic(
		position::Float64, boundary::Float64, variance::Float64, magnitude::Float64, 
		time_step::Float64, driving::Float64, time::Float64, trajectory::Array{Float64})
	steps = convert(Int64, ceil(time / time_step))
	trajectory[1] = position
	for step = 1:steps
		noise = Random.randn()
		position += variance * noise
		force = magnitude * sin(position) + driving
		position += time_step * force
		position -= boundary * floor(position / boundary)
		trajectory[step] = position
	end
	return nothing
end

state = PeriodicParticle(0, 2pi)
dynamics = PeriodicallyDrivenBrownian(0.001, 1, 2, 1)
time = 100.0
steps = evolutions(dynamics, time)
trajectory = ParticleTrajectory(steps)
print("Generic")
@time evolve!(state, dynamics, time, trajectory)
print("Generic")
@time evolve!(state, dynamics, time, trajectory)

print("With objects")
@time periodic_driven_brownian_trajectory(state, dynamics, time, trajectory)
print("With objects")
@time periodic_driven_brownian_trajectory(state, dynamics, time, trajectory)

print("With primitives")
@time periodic_driven_brownian_trajectory_basic(0.0, 2pi, sqrt(2pi*0.001), 2.0, 0.001, 1.0, time, zeros(steps))
print("With primitives")
@time periodic_driven_brownian_trajectory_basic(0.0, 2pi, sqrt(2pi*0.001), 2.0, 0.001, 1.0, time, zeros(steps))

#Plots.default(legend = false)
#plt = Plots.plot()
#Plots.plot!(plt, trajectory.positions, 
#			linecolor = "black", linealpha = 1)
#Plots.display(plt)