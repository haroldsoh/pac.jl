# The POMCP online solver for discrete POMDP models

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

    pomcp.rolloutPolicy = defaultRolloutPolicy
    pomcp.searchPolicy = POUCT

    pomcp.tree = Dict{Vector, POMCPTreeNode}()

    return pomcp
  end

end


# online solver, returns action and value of action
# doAction is a callback function that will execute once the action is found
function solve!(model::POMDP, solver::POMCP, history, doAction::Function)

  (action, value) = search!(model::POMDP, solver::POMCP, history)
  # take the action
  (obs, reward) = doAction(action)

  # prune the tree
  newhistory = [history, action, obs]
  newroot = solver.tree[newhistory]
  pruneTree!(solver, newhistory)

  return (action, obs, newhistory, reward)
end


function search!(model::POMDP, solver::POMCP, history)
  for i=1:solver.num_loops
    state = Nothing()
    if isempty(history)
      state = sampleInitialState(model)
    else
      state = sampleStateFromBelief(solver, history)
    end
    @assert state != Nothing()
    simulate!(model, solver, state, history, solver.depth)
  end

  best_value = -Inf
  best_action = Nothing()
  for a in model.actions()
    aug_history = [history, a]
    value = solver.tree[aug_history].value
    if (value > best_value) || (value == best_value && (rand() > 0.5))
      best_value = value
      best_action = a
    end
  end

  @assert best_action != Nothing()

  return (best_action, best_value)
end

function simulate!(model::POMDP, solver::POMCP, state, history, depth)
  if depth == 0
    return 0
  end

  if !haskey(solver.tree, history)
    solver.tree[history] = POMCPTreeNode()
    for action in model.actions()
      aug_history = [history, action]
      solver.tree[aug_history] = POMCPTreeNode()
    end
    return rollout!(model, solver, state, history, depth)
  end

  action = solver.searchPolicy(model, solver, history)
  newstate = model.transition(state, action)
  obs = model.emission(newstate)

  reward = model.reward(state, action, newstate)
  newhistory = [history, action]
  # recursive call until depth reached
  reward += model.discount*simulate!(model, solver,
                                       newstate, [newhistory, obs],
                                       depth-1)

  node = solver.tree[history]
  PAC.incrementBelief(node.belief, state, 1)
  node.count += 1

  childnode = solver.tree[newhistory]
  childnode.count += 1
  childnode.value = childnode.value + (reward - childnode.value)/childnode.count
  return reward
end

function sampleInitialState(model::POMDP)
  return PAC.sampleStateFromBelief(model.initialStateDist())
end

function sampleStateFromBelief(solver::POMCP, history)
  belief = solver.tree[history].belief
  return PAC.sampleStateFromBelief(belief)
end

# generate the next state
function generate(model::POMDP, state, action)
  next_state = model.transition(state, action)
  obs = model.emission(next_state)
  reward = model.reward(state, action, next_state)
  return (next_state, obs, reward)
end

function rollout!(model::POMDP, solver::POMCP, state, history, depth::Int64)
  # check if we have reached required search depth
  if depth == 0
    return 0
  end

  action = solver.rolloutPolicy(model, state, history )
  next_state, obs, reward = generate(model, state, action)

  if model.isTerminal(next_state)
    return reward
  else
    aug_history = [history, obs, action]
    return reward +
      model.discount*rollout!(model, solver, next_state, aug_history, depth-1)
  end
end

############################################
# Rollout and Search Policies
############################################
function defaultRolloutPolicy(model::POMDP, state, history)
  actions = model.actions()
  return actions[rand(1:end)]
end

function POUCT(model::POMDP, solver::POMCP, history)
  best_action = Nothing()
  best_value = -Inf

  actions = model.actions()

  log_total_count = log(solver.tree[history].count)
  if log_total_count == 0
    # if no counts yet, just return a random action
    return actions[rand(1:end)]
  end

  # select best action based on UCT search
  for action in actions
    aug_history = [history, action]
    aug_count = solver.tree[aug_history].count #this can be zero

    aug_value = Inf
    if aug_count > 0
      aug_value = solver.tree[aug_history].value +
        solver.c_tradeoff*sqrt(log_total_count/aug_count)
    end
    if aug_value > best_value || (aug_value == best_value && (rand() > 0.5))
      best_value = aug_value
      best_action = action
    end
  end
  #println((history[end], best_action, best_value))
  @assert best_action != Nothing()

  return best_action
end


# Helper functions
function resetTree!(solver::POMCP)
  solver.tree = Dict{Vector, POMCPTreeNode}()
end

function pruneTree!(solver::POMCP, history)
  for (key, val) in solver.tree
    # check if key prefix matches history
    if length(key) < length(history)
      delete!(solver.tree, key)
      continue
    end

    for (e,h) in zip(key, history)
      if e != h
        delete!(solver.tree, key)
        break
      end
    end
  end

end
