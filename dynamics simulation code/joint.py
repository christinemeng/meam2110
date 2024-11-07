import numpy as np

class JointType():
  '''
  An enum different joint types. Useful for tracking the type of a joint.

  To facilitate autograding, we do not use the `Enum` class
  '''
  fixed = 1
  rotation = 2
  translation = 3

class UnitVector():
  '''
  Define canonical x, y, z axes. Usesful to not have to always write them by hand
  '''
  x = np.array([1,0,0])
  y = np.array([0,1,0])
  z = np.array([0,0,1])

class Joint(object):
  '''
  A joint that connects to bodies or frames
  Joints are defined by a JointType (fixed, rotation, or translation)

  For rotation and translation joints are defined by an axis, which
  describes allowable rotational or translational motion. Rotation
  about the axis/translation along the axis. Fixed joints do not
  specify an axis.

  Joints define a transformation between the parent frame P and the
  child frame C, along with the relative position between the points
  Po and Co.

  There is also an intermediate joint frame, J, with point Jo.

  The relationships between P and J, and Po and Jo, are defined via
  fixed terms R_P_J and r_Po_Jo (expressed in P coordinates).

  The relationship between J and C is determined by the joint itself,
  for example, a rotation about the jx axis by a given angle, or
   a translation along the jz axis.

  Joints have the following properties:

    - parent: The parent body/frame P
    - child: The child body/frame C
    - type: A `JointType` enum
    - axis: A `UnitVector` axis of rotation or translation, expressed in J coordinates,
    - R_P_J: The rotation matrix from P to J, \( ^P R^J \)
    - r_Po_Jo: Expressed in P coordinates, \( ^{P_o}\\vec r^{J_o} \)
  '''

  def __init__(self, parent, child, type, axis, r_Po_Jo = np.zeros(3), R_P_J = np.eye(3)):
    '''
    The constructor method
  
    Args:
      parent: The parent body/frame P
      child: The child body/frame C
      type: The JointType
      axis: The axis of rotation or translation, expressed in J
            coordinates. Note that the axis must be one a `UnitVector`, one of
            [1;0;0], [0;1;0], or [0;0;1]
      r_Po_Jo: The position of the joint origin from the parent origin,
              expressed in P coordinates, \( ^{P_o}\\vec r^{J_o} \)
      R_P_J: The rotation matrix from P to J, \( ^P R^J \)
    '''

    # % TODO Check input values for compliance
    # if type ~= JointType.Fixed && ~isequal([1;0;0], axis) ... 
    #     && ~isequal([0;1;0], axis) && ~isequal([0;0;1], axis)
    #   error('Joint axis must be [1;0;0], [0;1;0], or [0;0;1]');
    # end

    # if ~isequal(size(r_Po_Jo), [3 1])
    #   error('r_Po_Jo must be 3x1');
    # end

    # if ~isequal(size(R_P_J), [3 3])
    #   error('R_P_J must be 3x3');
    # end

    self.parent = parent
    self.child = child
    self.type = type
    self.axis = axis
    self.R_P_J = R_P_J
    self.r_Po_Jo = r_Po_Jo
    self.name = parent.name + '_' + child.name
    child.SetParentJoint(self)
    parent.AddChildJoint(self)