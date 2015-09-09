discount_detm = 0.99
states_detm = [1,2,3,4,5]
actions_detm = states_detm

function initialStateDist_detm()
  return BeliefParticles(states_detm, [0.2,0.2,0.2,0.2,0.2])
end

function emission_detm(state, action, next_state)
  return next_state
end

function transition_detm(state, action)
  return states_detm[rand(1:end)]
  @assert false
end

function reward_detm(state, action, next_state)
  return state == action? 1 : -1
end

function isTerminal_detm(state)
  return false
end

function getActions_detm()
  return actions_detm
end

# create a new POMDP
detm_problem = POMDP( initialStateDist_detm,
                    getActions_detm,
                    emission_detm,
                    reward_detm,
                    transition_detm,
                    isTerminal_detm,
                    discount_detm)
