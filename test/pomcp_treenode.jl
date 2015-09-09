using PAC
using Base.Test

include("testutils.jl")

# Tests
println("-------- POMCP Tree Test --------")
t = PAC.POMCPTreeNode()

keys = (
  ["a1"],
  ["a1", "o1"],
  ["a2"],
  ["a2", "o3"],
  ["a3", "o1", "a1"]
)

values = [12, 35, 24, 56, 125]

for i = 1:5
  newnode = PAC.POMCPTreeNode(nothing, values[i])
  t[keys[i]] = newnode
end

for i = 1:5
  @test t[keys[i]].value == values[i]
end

@test PAC.haskey(t, keys[1])
@test !PAC.haskey(t, ["a0", "o2"])

print("- POMCP Tree Test: ")
print_with_color(:green, "PASSED\n")
