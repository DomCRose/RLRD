import Random
import Plots

abstract type State end
abstract type Particle <: State end
abstract type Trajectory end
abstract type Dynamics end
abstract type ParameterizedDynamics <: Dynamics end
abstract type Algorithm end

mutable struct PeriodicParticle <: Particle
	position::Float64
	boundary::Float64
end


mutable struct ParticleTrajectory <: Trajectory
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


mutable struct FourierDrivenBrownian <: ParameterizedDynamics
	time_step::Float64
	diffusion_coefficient::Float64
	variance::Float64
	dimension::Int8
	basis_index::Array{Float64}
	parameters::Array{Float64}
	current_update::Array{Float64}
end
function FourierDrivenBrownian(time_step, diffusion_coefficient, dimension)
	dynamics = FourierDrivenBrownian(
		time_step, diffusion_coefficient, sqrt(time_step*diffusion_coefficient),
		dimension, collect(1:dimension), zeros(2dimension + 1), zeros(2dimension + 1))
end
function evolve!(state::PeriodicParticle, dynamics::FourierDrivenBrownian)
	noise = Random.randn()
	state.position += dynamics.variance * noise
	force = dynamics.magnitude * sin(state.position) + dynamics.driving
	state.position += dynamics.time_step * force
	state.position -= state.boundary * floor(state.position / state.boundary)
	return nothing
end

struct DifferentialActorCritic <: Algorithm
	learning_rate::Float64
	average_reward::Float64
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




state = PeriodicParticle(0, 2pi)
dynamics = PeriodicallyDrivenBrownian(0.001, 1, 2, 1)
time = 10000.0
steps = evolutions(dynamics, time)
print(steps)
trajectory = ParticleTrajectory(steps)
print("Generic")
@time evolve!(state, dynamics, time, trajectory)
print("Generic")
@time evolve!(state, dynamics, time, trajectory)

#Plots.default(legend = false)
#plt = Plots.plot()
#Plots.plot!(plt, trajectory.positions, 
#			linecolor = "black", linealpha = 1)
#Plots.display(plt)