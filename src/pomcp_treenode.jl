
type POMCPTreeNode
  keyelem # key element associated with this node
  value::Float64
  count::Int64  # serves as total count when history does not end with an action
                # and action dependent count when history ends with an action
  belief::BeliefParticles # belief states

  # child nodes
  children::Dict{Any, POMCPTreeNode}
end

# constructor for no arguments
POMCPTreeNode() = POMCPTreeNode(nothing)
POMCPTreeNode(keyelem) = POMCPTreeNode(keyelem, 0.0)
POMCPTreeNode(keyelem, value) = POMCPTreeNode(keyelem, value, 0.0)
POMCPTreeNode(keyelem, value, count) = POMCPTreeNode(keyelem, value, count, BeliefParticles(), Dict{Any, POMCPTreeNode}())

# adds a new child node and returns the new child node
function addChild!(node::POMCPTreeNode, keyelem)
  newnode = POMCPTreeNode(keyelem)
  node.children[keyelem] = newnode
  return newnode
end

# removes a child node
function rmChild!(node::POMCPTreeNode, keyelem)
  delete!(node.children, keyelem)
end

function haskey(root::POMCPTreeNode, key::Any)
  node = root
  for k in key
    if !Base.haskey(node.children, k)
      return false
    else
      node = node.children[k]
    end
  end
  return true
end

# gets a node with a specfic key
# if no node associated with key, a new one is created
function getNode!(root::POMCPTreeNode, key)
  node = root
  for k in key
    if !Base.haskey(node.children, k)
      addChild!(node, k)
    end
    node = node.children[k]
  end
  return node
end

function getindex(root::POMCPTreeNode, key)
  node = root
  for k in key
    node = node.children[k]
  end
  return node
end

function setindex!(root::POMCPTreeNode, snode::POMCPTreeNode, key)
  node = getNode!(root, key)
  transferMembers!(snode, node)
  if length(key) > 0
    node.keyelem = key[end]
  else
    node.keyelem = nothing
  end
  return node
end

function transferMembers!(source::POMCPTreeNode, dest::POMCPTreeNode)
  dest.keyelem = source.keyelem # key element associated with this node
  dest.value = source.value
  dest.count = source.count
  dest.belief = source.belief
  dest.children = source.children
end
