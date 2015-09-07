# POMCP solver test

using PAC
using Base.Test

include("../pomdp.jl") #include pomdp test file for sample pomdp

depth = 5
c_tradeoff = 1.0
rolloutPolicy = defaultRolloutPolicy
numloops = 1000
stop_eps = 1e-3
seed = 0

# create our solver
pomcp = POMCP(depth = depth,
  c_tradeoff = c_tradeoff,
  rolloutPolicy = defaultRolloutPolicy,
  numloops = numloops,
  stop_eps = 1e-3,
  seed = 0)

# test the sampler

# test the rollout policy
