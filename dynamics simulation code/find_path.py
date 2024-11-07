def FindPath(A, B):
  '''
  Determine the path through the kinematic tree from body/frame A to B

  Args:
    A: Body A (the start of the path)
    B: Body B (the end of the path)

  Returns:
    path: A list of the bodies along the chain. The first element of the
           path will always be A, and the last will be B
    directions: A vector corresponding to the joints along the path. This
                 vector is 1 shorter than the path vector, and iss a vector of
                 signs (+1 or -1), indicating whether traversing the path goes
                 in the +1 "normal direction", from parent to child, or in the
  '''
  success, path, directions = PathDepthFirstSearch(B, [A], [])
  assert(success), "Failed to find a path between " + A.name + " and " + B.name + ". Double-check that both bodies have been added to the system."
  return path, directions


def PathDepthFirstSearch(B, path, directions):
  '''
  Recursive depth first search for the path from A to B
  '''
  if path[-1] == B:
    return True, path, directions

  # Expand the parent of the last body on the path, assuming the previous
  # direction wasn't parent->child
  parent = path[-1].parent
  if parent and (len(directions) == 0 or directions[-1] == -1):
    path_parent = path.copy()
    path_parent.append(parent)
    directions_parent = directions.copy()
    directions_parent.append(-1)

    [success, path_parent, directions_parent] = PathDepthFirstSearch(B, path_parent, directions_parent)

    if success:
      return True, path_parent, directions_parent

  # Expand all chldren
  for child in path[-1].children:
    if len(path) < 2 or child is not path[-2]:
      path_child = path.copy()
      path_child.append(child)
      directions_child = directions.copy()
      directions_child.append(1)

      [success, path_child, directions_child] = PathDepthFirstSearch(B, path_child, directions_child)

      if success:
        return True, path_child, directions_child


  return False, path, directions