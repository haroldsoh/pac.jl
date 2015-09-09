# test POMDP
using PAC
using Base.Test

seed = 100
srand(seed)

include("testutils.jl")
include("spouse_pomdp.jl") # creates a pomdp named my_problem

policy = safePolicy
curr_state = "happy"

obs_freq = Dict(observations, zeros(size(observations)))
state_freq = Dict(states, zeros(size(states)))
total_reward = 0.0

# simulate for a million trials
N =1_000_000
obs = Nothing()
for i=1:N
  old_state = curr_state
  action = policy(obs, actions)
  curr_state = my_problem.transition(curr_state, action)
  total_reward += my_problem.reward(old_state, action, curr_state)
  obs = my_problem.emission(old_state, action, curr_state)

  # collect frequencies
  obs_freq[obs] += 1
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
safe_obs_freq = ["calm"=>258116.0,"smiling"=>643698.0,"yelling"=>98186.0]
safe_state_freq = ["angry"=>195917.0,"happy"=>804083.0]

if seed == 100
  errortol = 1e-9*N
else
  errortol = 1e-2*N
end

testDict(obs_freq, safe_obs_freq, errortol)
testDict(state_freq, safe_state_freq, errortol)
