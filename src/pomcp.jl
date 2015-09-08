# The POMCP online solver for discrete POMDP problems

type POMCP <: PACSolver
  actions
  tree::Dict{Any, Any}
  counts::Dict{Any, Int64}
  total_counts::Dict{Any, Int64}
  value::Dict{Any,Float64}
  belief::Dict{Any, Float64}

  num_particles::Int64 # number particules for belief state

  depth::Int64        # search depth
  c_tradeoff::Float64 # exploration/exploitation trade-off

  rolloutPolicy::Function
  searchPolicy::Function

  num_loops::Int64
  stop_eps::Float64

  function POMCP(; actions = [],
    rolloutPolicy::Function = defaultRolloutPolicy,
    searchPolicy::Function = POUCT,
    depth::Int64 = 5,
    c_tradeoff::Float64 = 1.0,
    num_particles::Int64 = 1000,
    num_loops::Int64 = 10_000,
    stop_eps::Float64 = 1e-3
    )
    pomcp = new()

    pomcp.actions = actions
    pomcp.depth = depth
    pomcp.c_tradeoff = c_tradeoff
    pomcp.rolloutPolicy = rolloutPolicy
    pomcp.num_particles = num_particles
    pomcp.num_loops = num_loops
    pomcp.stop_eps = stop_eps

    pomcp.tree = Dict{Any, Any}()
    pomcp.counts = Dict{Any, Int64}()
    pomcp.total_counts = Dict{Any, Int64}()
    pomcp.value = Dict{Any,Float64}()
    pomcp.belief = Dict{Any, Int64}()

    return pomcp
  end

end

# online solver, returns action and value of action
function solve!(problem::POMDP, solver::POMCP, history)

end

function search!(problem::POMDP, solver::POMCP, history)
  if isempty(history)
    state = sampleInitialState(solver::POMCP)
  else
    state = sampleStateFromBelief(solver::POMCP)
  end



end

function sampleStateFromBelief(solver::POMCP)
  r = rand()
  cumsum = 0.0
  freqsum = sumBeliefFreq(solver)
  last_state = 0.0
  for (state, state_freq) in solver.belief
      cumsum += state_freq/freqsum
      if r < cumsum
          return state
      end
      last_state = state
  end
  return last_state
end

function sumBeliefFreq(solver::POMCP)
  freqsum = 0.0
  for (state, state_freq) in solver.belief
    freqsum += state_freq
  end
  return freqsum
end

# generate the next state
function generate(problem::POMDP, state, action)
  next_state = problem.transition(state, action)
  obs = problem.emission(next_state)
  reward = problem.reward(state, action, next_state)
  return (next_state, obs, reward)
end

function rollout!(problem::POMDP, solver::POMCP, state, history, depth::Int64)
  # check if we have reached required search depth
  if depth == 0
    return 0
  end

  action = solver.rolloutPolicy(problem, history, state, solver.actions)
  next_state, obs, reward = generate(problem, state, action)

  if problem.isTerminal(next_state)
    return reward
  else
    push!(history, (observation, action))
    return reward +
      problem.discount*rollout(problem, solver, next_state, history, depth-1)
  end
end

function defaultRolloutPolicy(problem::POMDP, history, state, actions)
  return actions[rand(1:end)]
end

function POUCT(solver::POMCP, history, actions)

end
