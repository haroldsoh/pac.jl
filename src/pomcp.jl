# The POMCP online solver for discrete POMDP problems

# Define our tree node types
typealias  BeliefParticles Dict{Any, Float64} # from states to frequency counts

type POMCPTreeNode
  value::Float64
  count::Int64  # serves as total count when history does not end with an action
                # and action dependent count when history ends with an action
  belief::BeliefParticles # belief states
end

POMCPTreeNode() = POMCPTreeNode(0.0, 0.0, BeliefParticles())


# POMCP Solver
type POMCP <: PACSolver
  tree::Dict{Vector, POMCPTreeNode}
  num_particles::Int64 # max number particles for belief state

  depth::Int64        # search depth
  c_tradeoff::Float64 # exploration/exploitation trade-off

  rolloutPolicy::Function
  searchPolicy::Function

  num_loops::Int64
  stop_eps::Float64

  function POMCP(;
    rolloutPolicy::Function = defaultRolloutPolicy,
    searchPolicy::Function = POUCT,
    depth::Int64 = 5,
    c_tradeoff::Float64 = 1.0,
    num_particles::Int64 = 1000,
    num_loops::Int64 = 1000,
    stop_eps::Float64 = 1e-3
    )
    pomcp = new()

    pomcp.depth = depth
    pomcp.c_tradeoff = c_tradeoff
    pomcp.rolloutPolicy = rolloutPolicy
    pomcp.num_particles = num_particles
    pomcp.num_loops = num_loops
    pomcp.stop_eps = stop_eps

    pomcp.tree = Dict{Vector, POMCPTreeNode}()

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

function simulate!(problem::POMDP, solver::POMDP, history, depth)
  if depth == 0
    return 0
  end

  if !haskey(solver.tree, history)
    solver.tree[history] = POMCPTreeNode(0.0, 0.0, BeliefParticles() )
    for action in problem.actions()
      newhistory = push(history, action)
      solver.tree[newhistory] = POMCPTreeNode(0.0, 0.0, BeliefParticles() )
    end
  end

end


function sampleStateFromBelief(solver::POMCP, history)
  r = rand()
  cumsum = 0.0
  belief = solver.tree[history].belief
  freqsum = sumBeliefFreq(belief)
  last_state = 0.0
  for (state, state_freq) in belief
      cumsum += state_freq/freqsum
      if r < cumsum
          return state
      end
      last_state = state
  end
  return last_state
end

function sumBeliefFreq(belief)
  freqsum = 0.0
  for (state, state_freq) in belief
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

function rollout(problem::POMDP, solver::POMCP, state, history, depth::Int64)
  # check if we have reached required search depth
  if depth == 0
    return 0
  end

  action = solver.rolloutPolicy(problem, history, state)
  next_state, obs, reward = generate(problem, state, action)

  if problem.isTerminal(next_state)
    return reward
  else
    aug_history = push(history, observation)
    push!(aug_history, action)
    return reward +
      problem.discount*rollout(problem, solver, next_state, aug_history, depth-1)
  end
end

function defaultRolloutPolicy(problem::POMDP, history, state)
  actions = problem.actions()
  return actions[rand(1:end)]
end

function POUCT(problem::POMDP, solver::POMCP, history)
  best_action = Nothing()
  best_value = -Inf
  log_total_count = log(solver.tree[history].count)
  for action in problem.actions()
    aug_history = push(history, action)
    aug_count = solver.tree[aug_history].count #this can be zero
    aug_value = solver.tree[aug_history] +
      solver.c_tradeoff*sqrt(log_total_count/aug_count)
  end
end
