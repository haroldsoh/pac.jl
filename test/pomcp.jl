# POMCP solver test

using PAC
using Base.Test

include("testUtils.jl")
include("samplePomdp.jl") #include pomdp test file for sample pomdp

depth = 5
c_tradeoff = 1.0
rolloutPolicy = PAC.defaultRolloutPolicy
num_loops = 1000
stop_eps = 1e-3
seed = 0
srand(seed)

# create our solver
pomcp = POMCP(depth = depth,
  c_tradeoff = c_tradeoff,
  rolloutPolicy = PAC.defaultRolloutPolicy,
  searchPolicy = PAC.POUCT,
  num_loops = num_loops,
  stop_eps = stop_eps)

###########################################
# test the belief sampler
###########################################
history = ["a1", "o1"]
pomcp.tree[history] = PAC.POMCPTreeNode()
belief = pomcp.tree[history].belief
belief["a"] = 0.5
belief["b"] = 0.3
belief["c"] = 0.2

state_freq = Dict(["a", "b", "c"], [0.0 ,0.0, 0.0])
N = 1_000_000
for i=1:N
  state_freq[PAC.sampleStateFromBelief(pomcp, history)] += 1.0
end
for (state,freq) in state_freq
  state_freq[state] /= N
end
testDict(state_freq, belief, 1e-2)
print_with_color(:green, "Belief Sampler Test: [PASSED]\n")

###########################################
# test using on my_problem
###########################################
type Simulator
  state
  model::POMDP
end

function doActionCallback!(action, sim::Simulator, model::POMDP)
  next_state = model.transition(sim.state, action)
  r = model.reward(sim.state, action, next_state)
  obs = model.emission(next_state)
  sim.state = next_state
  return (obs, r)
end

sim = Simulator("happy", my_problem)
doActionCallback(action) = doActionCallback!(action, sim, my_problem)

resetTree!(pomcp)
N = 1000
total_reward = 0.0
history = []
for i=1:N
  (action, obs, newhistory, r) = solve!(my_problem, pomcp, history, doActionCallback)
  total_reward += r
  print(i)
  print(": ")
  println(total_reward)
end

println(total_reward)
