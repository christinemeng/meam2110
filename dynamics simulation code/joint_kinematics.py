from . import UnitVector, JointType
import numpy as np
import sympy as sym

def JointTransformation(system, joint, q):
  '''
  Compute the transformation across a joint J, from parent P to child C.
  Note that this method must compute the transformation from P to J, and
  then J to C.

  Args:
    joint: The Joint object
    q: The configuration vector (typically, vector of joint angles/positions)

  Returns:
    r_Po_Co: The position vector from Po to Co, expressed in P coordinates, \( ^{P_o} \\vec r^{C_o} \)
    R_P_C: The rotation matrix from P to C, \( ^P R^C \)
  '''
  # Use the sympy or math libraries for trig functions, depending on if the input is a floating point
  # number or a symbolic variable
  if isinstance(q[0], sym.Expr):
    from sympy import sin, cos
  else:
    from math import sin, cos

  if joint.type == JointType.fixed:
    # If the joint is a fixed type, then the parent and child are welded together
    r_Po_Co = joint.r_Po_Jo
    R_P_C = joint.R_P_J
  elif joint.type == JointType.rotation:
    # Get the angle of the relevant joint
    angle = q[system.joints[joint]]

    # YOUR CODE GOES BELOW (PART 1)
    # Calculate R_J_C for each possible axis of rotation
    if np.array_equal(joint.axis, UnitVector.x):
      R_J_C = np.array([[1,0,0],
                       [0,cos(angle),-1*sin(angle)],
                       [0,sin(angle),cos(angle)]])
  
    elif np.array_equal(joint.axis, UnitVector.y):
      R_J_C = np.array([[cos(angle),0,sin(angle)],
                       [0,1,0],
                       [-1*sin(angle),0,cos(angle)]])
      
    elif np.array_equal(joint.axis, UnitVector.z):
      R_J_C = np.array([[cos(angle),-1*sin(angle),0],
                       [sin(angle),cos(angle),0],
                       [0,0,1]])

    # For a rotation joint, the child origin is located at the joint origin
    r_Po_Co = joint.r_Po_Jo

    # Calculate R_P_C
    # YOUR CODE GOES HERE (PART 1)
    R_P_C = joint.R_P_J @ R_J_C

  elif joint.type == JointType.translation:
    # Get the position of the relative joint
    position = q[system.joints[joint]]

    # Calculate R_P_C and r_Po_Co
    # YOUR CODE GOES HERE
    r_Po_Co = joint.axis * position
    R_P_C = joint.R_P_J

  return r_Po_Co, R_P_C

def JointChildVelocity(system, joint, qdot):
  '''
  Compute the linear and angular velocities from joint parent to child

  Args:
    system: The RigidBodySystem
    joint: The relevant Joint object
    qdot: A vector of all joint velocities

  Returns:
    v_P_Co: the linear velocity of child origin Co in parent P (in P coordinates), \( ^P \\vec v^{C_o} \)
    w_P_C: the angular velocity of child C in parent P (in P coordinates), \( ^P \\vec \omega^C \)
  '''
  # YOUR CODE needed to calculate the velocity and angular velocities below
  if joint.type == JointType.fixed:
    v_P_Co = np.zeros(3)
    w_P_C = np.zeros(3)
  elif joint.type == JointType.rotation:
    # extract the scalar velocity associated with this joint
    angle_dot = qdot[system.joints[joint]]

    v_P_Co = np.zeros(3)
    w_P_C = joint.R_P_J @ (angle_dot * joint.axis)
  elif joint.type == JointType.translation:
    # extract the scalar velocity associated with this joint
    position_dot = qdot[system.joints[joint]]

    v_P_Co = joint.R_P_J @ (position_dot * joint.axis)
    w_P_C = np.zeros(3) # replace me!

  return v_P_Co, w_P_C

def JointChildAcceleration(system, joint, qddot):
  '''
  Compute the linear and angular accelerations from joint parent to child

  Args:
    system: The RigidBodySystem
    joint: The relevant Joint object
    qdot: A vector of all joint accelerations

  Returns:
    a_P_Co : the linear acceleration of child origin Co in parent P (in P coordinates), \( ^P \\vec a^{C_o} \)
    alpha_P_C : the angular acceleration of child C in parent P (in P coordinates), \( ^P \\vec \\alpha^C \)
  '''
  # YOUR CODE needed to calculate the velocity and angular velocities below
  if joint.type == JointType.fixed:
    a_P_Co = np.zeros(3)
    alpha_P_C = np.zeros(3)
  elif joint.type == JointType.rotation:
    # extract the scalar acceleration associated with this joint
    angle_ddot = qddot[system.joints[joint]]

    a_P_Co = np.zeros(3) 
    alpha_P_C = joint.R_P_J @ (angle_ddot * joint.axis)
  elif joint.type == JointType.translation:
    # extract the scalar velocity associated with this joint
    position_ddot = qddot[system.joints[joint]]
    
    a_P_Co = joint.R_P_J @ (position_ddot * joint.axis)
    alpha_P_C = np.zeros(3)

  return a_P_Co, alpha_P_C
