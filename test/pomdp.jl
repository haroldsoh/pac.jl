# test POMDP
using PAC
using Base.Test

seed = 100
srand(seed)

include("testUtils.jl")
include("samplePomdp.jl") # creates a pomdp named my_problem

policy = safePolicy
curr_state = "happy"

obs_freq = Dict(observations, zeros(size(observations)))
state_freq = Dict(states, zeros(size(states)))
total_reward = 0.0

# simulate for a million trials
N =1_000_000
for i=1:N
  obs = my_problem.emission(curr_state)
  obs_freq[obs] += 1
  old_state = curr_state
  action = policy(obs, actions)
  curr_state = my_problem.transition(curr_state, action)
  total_reward += my_problem.reward(old_state, action, curr_state)
  state_freq[curr_state] += 1
end

print("Total Reward: ")
println(total_reward)
println("Simulation Frequencies")
print("Observations: ")
println(obs_freq)
print("States: ")
println(state_freq)

# safe policy values
safe_obs_freq = ["calm"=>258555.0,"smiling"=>643653.0,"yelling"=>97792.0]
safe_state_freq = ["angry"=>195798.0,"happy"=>804202.0]

if seed == 100
  errortol = 1e-9*N
else
  errortol = 1e-2*N
end

testDict(obs_freq, safe_obs_freq, errortol)
testDict(state_freq, safe_state_freq, errortol)
