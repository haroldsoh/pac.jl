# POMCP solver test

using PAC
using Base.Test

include("testutils.jl")
include("detm_pomdp.jl")

# options
test_belief_sampler = false
test_detm_pomdp = true # deterministic POMDP (MDP) test

# POMCP params
depth = 7
c_tradeoff = 0.5
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
if test_belief_sampler
  history = ["a1", "o1"]
  pomcp.tree[history] = PAC.POMCPTreeNode()
  belief = pomcp.tree[history].belief
  belief["a"] = 0.5
  belief["b"] = 0.3
  belief["c"] = 0.2

  state_freq = Dict(["a", "b", "c"], [0.0 ,0.0, 0.0])
  N = 1_000_000
  @time for i=1:N
    state_freq[PAC.sampleStateFromBelief(pomcp, history)] += 1.0
  end
  for (state,freq) in state_freq
    state_freq[state] /= N
  end
  testDict(state_freq, belief, 1e-2)
  print("Belief Sampler Test: ")
  print_with_color(:green, "PASSED\n")
end

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
  obs = model.emission(sim.state, action, next_state)
  sim.state = next_state
  return (obs, r)
end

if test_detm_pomdp
  println("-------- Running Deterministic POMDP Test --------")
  initial_state = PAC.sampleStateFromBelief(detm_problem.initialStateDist())
  sim = Simulator(initial_state, detm_problem)
  doActionCallback(action) = doActionCallback!(action, sim, detm_problem)

  resetTree!(pomcp)
  N = 100
  total_reward = 0.0

  prev_obs = Nothing()

  @time for i=1:N
    prev_state = sim.state
    (action, obs, r) = solve!(detm_problem, pomcp, doActionCallback)
    total_reward += r
    println((i, prev_state, prev_obs, action, total_reward))
    prev_obs = obs
  end

  @test total_reward == N # because you won't know the initial state
  print("- Deterministic POMDP (MDP) Test: ")
  print_with_color(:green, "PASSED\n")

end
