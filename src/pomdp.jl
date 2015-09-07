type Pomdp <: PACModel
  states
  actions
  obs # observations
  emission::Function # emission function (from states to observations)
  reward::Function
  transition::Function
  discount::Float64
end
