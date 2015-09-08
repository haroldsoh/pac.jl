# POMDP model
# Note that states, actions and observations are specified implicitly through
#   associated functions. This is done so that we never have to explicitly
#   store very large state, action and observation spaces.

type POMDP <: PACModel
  actions::Function  # actions() returns list of possible actions
  emission::Function # emission(state) returns observation
  reward::Function   # reward(state, action, next_state) returns Float64
  transition::Function # transition(state, action) returns next_state
  isTerminal::Function #isTerminal(state) returns true if state is terminal state
  discount::Float64
end
