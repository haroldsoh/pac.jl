# The POMCP online solver for discrete POMDP problems

type POMCP <: PACSolver
  Actions
  Tree::Dict{Any, Any}
  Counts::Dict{Any, Int64}
  TotalCounts::Dict{Any, Int64}
  Value::Dict{Any,Float64}
  Belief::Dict{Any, Int64}

  num_particles::Int64 # number particules for belief state

  depth::Int64        # search depth
  c_tradeoff::Float64 # exploration/exploitation trade-off

  rolloutPolicy::Function
  searchPolicy::Function

  num_loops::Int64
  stop_eps::Float64

  seed::Int64

  function POMCP(;
    actions = [],
    rolloutPolicy::Function = defaultRolloutPolicy,
    searchPolicy::Function = POUCT,
    depth::Int64 = 5,
    c_tradeoff::Float64 = 1.0,
    num_particles = 1000,
    num_loops::Int64 = 10_000,
    stop_eps::Float64 = 1e-3,
    seed::Int64 = 0
    )
    pomcp = new()

    pomcp.actions = actions
    pomcp.depth = depth
    pomcp.c_tradeoff = c_tradeoff
    pomcp.rolloutPolicy = rolloutPolicy
    pomcp.num_particles = num_particles
    pomcp.num_loops = num_loops
    pomcp.stop_eps = stop_eps

    pomcp.seed = seed
    srand(seed)

    return pomcp
  end

end

# online solver, returns action and value of action
function solve!(problem::POMDP, solver::POMCP, history)

end

function search!(problem::POMDP, solver::POMCP, history)

end

function sampleStateFromBelief(solver::POMCP)
  r = rand()
  cumsum = 0.0
  last_state = 0.0
  for (state, pstate) in solver.belief
      cumsum += pstate
      if r < cumsum
          return state
      end
      last_state = state
  end
  return last_state
end

function normalizeBelief(solver::POMCP)

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

  action = solver.rolloutPolicy(problem, state, solver.actions)
  next_state, obs, reward = generate(problem, state, action)

  if problem.isTerminal(next_state)
    return reward
  else
    push!(history, (observation, action))
    return reward +
      problem.discount*rollout(problem, solver, next_state, history, depth-1)
  end
end

function defaultRolloutPolicy(problem::POMDP, state, actions)
  return actions[rand(1:end)]
end

function POUCT(actions)

end
