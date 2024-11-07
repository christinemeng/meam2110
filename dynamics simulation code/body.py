import numpy as np
'''
The Body class represents a rigid body or frame, depending on whether
it has mass or not

This class stores the inertial and visual propeties of the body "B."

Each body comes with two defined points, an origin Bo and the center of mass
Bcm.

All units are in SI

Properties:
- name (string)
- mass (double)
- I_B_Bcm The inertia matrix of body B about the center of mass
          in B coordinates (2D array)
- r_Bo_Bcm The position of the center of mass, Bcm, with respect to Bo
           in B coordinates (1D array)
- body_vis The visual properties usd to draw the body
           (MeshcatBodyVisual object)
- parent_joint The joint to the parent body in the kinematic tree
- parent The parent body
- child_joints A list of joints to any children bodies in the kinematic tree
- children A list of child bodies
'''
class Body(object):
  def __init__(self, name, mass = 0, I_B_Bcm = np.zeros((3,3)), r_Bo_Bcm = np.zeros(3), vis = None):
    '''
    The basic constructor for a body

    Args:
      name: The string name of the body
      mass: The mass of the body
      I_B_Bcm: The inertia matrix of body B about the center of mass
                (in B coordinates)
      r_Bo_Bcm: The position of the center of mass, Bcmwith respect to the
                 origin of the B frame, Bo (in B coordinates)
      vis : Visualization description (optional)
    '''
    assert(mass >= 0), 'Mass must be non-negative'
    assert(I_B_Bcm.shape == (3,3)), 'I_B_Bcm must be 3x3'
    assert(r_Bo_Bcm.shape == (3,)), 'r_Bo_Bcm must be a length 3 vector'

    self.name = name
    self.mass = mass
    self.I_B_Bcm = I_B_Bcm
    self.r_Bo_Bcm = r_Bo_Bcm
    self.vis = vis
    self.parent_joint = None
    self.parent = None
    self.child_joints = []
    self.children = []

  def SetParentJoint(self, parent_joint):
    '''
    Sets the parent joint of this body, and the parent body accordingly

    Args:
      self: The `Body` object
      parent_joint: The `Joint` object for the parent. Will use this to determine the parent body.
    '''
    self.parent_joint = parent_joint
    self.parent = parent_joint.parent

  def AddChildJoint(self, child_joint):
    '''
    Adds a child joint and child body to the list of children.

    Args:
      self: The `Body` object
      child_joint: A `Joint` object for a child. Will use this to also add a child body.
    '''
    self.child_joints.append(child_joint)
    self.children.append(child_joint.child)

  def Draw(self, r_No_Bo, R_N_B):
    '''
    Draw the body in the given location

    Args:
    
      r_No_Bo: The position from the inertial origin No to the body origin Bo
    
      R_N_B: The rotation matrix from N to B
    '''
    if self.vis is not None:
      self.vis.Draw(r_No_Bo, R_N_B)
