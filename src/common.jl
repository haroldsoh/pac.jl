# from states to frequency counts
typealias  BeliefParticles Dict{Any, Float64}

function sampleStateFromBelief(belief::BeliefParticles)
  r = rand()
  cumsum = 0.0
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

function sumBeliefFreq(belief::BeliefParticles)
  freqsum = 0.0
  for (state, state_freq) in belief
    freqsum += state_freq
  end
  return freqsum
end

function incrementBelief(belief::BeliefParticles, state, count)
  try
      belief[state] = belief[state] + count
      catch error
         if isa(error, KeyError)
             belief[state] = count
         end
  end
end
