import numpy as np
from . import FindPath, JointTransformation, JointChildVelocity, JointChildAcceleration, BodyAngVelAndAccel, ChangeCoordinates

def PointKinematics(system, q, A, B, *, qdot = None, qddot = None, r_Ao_P = np.zeros(3), r_Bo_Q = np.zeros(3)):
  '''
  Calculate the kinematics of point P, where P is a point fixed on body A.
  All returned values in are in B coordinates.

  Usage:
    For some calculations, certain of the optional arguments are not needed. Four possible use cases are highlighted:

    1. Position only:

      <pre>r_Q_P_B, _, _ = PointKinematics(system, q, A, B, r_Ao_P = r_Ao_P, r_Bo_Q = r_Bo_Q)</pre>

    2. Velocity only:

      <pre>_, v_B_P, _ = PointKinematics(system, q, A, B, qdot = qdot, r_Ao_P = r_Ao_P)</pre>

    3. Position and velocity:

      <pre>r_Q_P_B, v_B_P, _ = PointKinematics(system, q, A, B, qdot = qdot, r_Ao_P = r_Ao_P, r_Bo_Q = r_Bo_Q)</pre>

    4. Position, velocity, and acceleration

      <pre>r_Q_P_B, v_B_P, a_B_P = PointKinematics(system, q, A, B, qdot = qdot, qddot = qddot, r_Ao_P = r_Ao_P, r_Bo_Q = r_Bo_Q)</pre>


  Args:
    system: The RigidBodySystem object
    q: vector of joint positions
    A: Body A, on which P is fixed
    B: Body B, on which Q is fixed

    The inputs below are optional, and must be entered by name. They default to zeros
  
    qdot: vector of joint velocities
    qddot: vector of joint accelerations
    r_Ao_P: The position of P from Ao, expressed in A coordinates
    r_Bo_Q: The position of Q from Bo, expressed in B coordinates

  Returns:
    r_Q_P_B: The position vector from Q to P, where Q is a point fixed on body B, \( ^Q \\vec r^P \)
    v_B_P: The linear velocity of P in B, \( ^B v^P \)
    a_B_P: The linear acceleration of P in B, \( ^B a^P \)
  '''
  if qdot is None:
    qdot = q * 0
  if qddot is None:
    qddot = q * 0

  path, directions = FindPath(B, A)

  # Call `C` the current child body along the path. Traversing the path, update until C is A
  # Initialize C = B
  C = B
  r_Bo_Co = np.zeros(3)
  v_B_Co = np.zeros(3)
  a_B_Co = np.zeros(3)

  for i, direction_i in enumerate(directions):
    # Set the new PARENT to be the old child
    r_Bo_Po = r_Bo_Co
    v_B_Po = v_B_Co
    a_B_Po = a_B_Co
    PARENT = C
    C = path[i+1]


    if direction_i > 0:
      JP = path[i]
      JC = path[i+1]
    else:
      JP = path[i+1]
      JC = path[i]
    joint = JC.parent_joint

    r_Po_Jo_J, R_P_C = JointTransformation(system, joint, q)
    v_Po_Jo_J, omega_J = JointChildVelocity (system, joint, qdot)
    a_Po_Jo_J, alpha_J = JointChildAcceleration(system, joint, qddot)

    if (direction_i > 0):
      r_Po_Jo_B = ChangeCoordinates(system, q, r_Po_Jo_J, PARENT, B)
      v_Po_Jo_B = ChangeCoordinates(system, q, v_Po_Jo_J, PARENT, B)
      a_Po_Jo_B = ChangeCoordinates(system, q, a_Po_Jo_J, PARENT, B)
      omega_B, alpha_B = BodyAngVelAndAccel(system, q, qdot, PARENT, B, qddot=qddot)
    else:
      r_Po_Jo_B = ChangeCoordinates(system, q, -1*r_Po_Jo_J, C, B)
      v_Po_Jo_B = ChangeCoordinates(system, q, -1*v_Po_Jo_J, C, B)
      a_Po_Jo_B = ChangeCoordinates(system, q, -1*a_Po_Jo_J, C, B)
      omega_B, alpha_B = BodyAngVelAndAccel(system, q, qdot, C, B, qddot=qddot)

    # YOUR CODE HERE
    # The goal of this block of code is to determine
    #  a) r_Bo_Co
    #  b) v_B_Co
    #  c) a_B_Co
    # for the new child (the next body along the path)
    #
    # Suggested approach:
    # 1) Calculate position/velocity/acceleration across the joint
    # 2) Calculate the position/velocity/acceleration from PARENT to C (keeping direction_i in mind)
    # 3) Calculate r_Bo_Co, v_B_Co, a_B_Co
    #
    # Throughout, be sure to keep track of what the coordinates used to express every vector!

    # Update position
    r_Bo_Co = r_Bo_Po + r_Po_Jo_B

    # Update velocity
    v_B_Co = v_B_Po + v_Po_Jo_B + np.cross(omega_B, r_Po_Jo_B) # replace me!

    # Update acceleration
    a_B_Co = a_B_Po + np.cross(alpha_B, r_Po_Jo_B) + np.cross(omega_B, np.cross(omega_B, r_Po_Jo_B)) + a_Po_Jo_B + np.cross(2*omega_B, v_Po_Jo_B) # replace me!

  # The last child is A
  r_Bo_Ao = r_Bo_Co
  v_B_Ao = v_B_Co
  a_B_Ao = a_B_Co

  # YOUR CODE HERE
  # We're not quite done. What we really want is the position/velocity/accleration of P, not Ao.
  sum_omegas, sum_alphas = BodyAngVelAndAccel(system, q, qdot, A, B, qddot=qddot)
  r_Ao_P_B = ChangeCoordinates (system, q, r_Ao_P, A, B)

  r_Q_P_B = r_Ao_P_B + r_Bo_Ao - r_Bo_Q

  v_B_P = v_B_Ao + np.cross(sum_omegas, r_Ao_P_B)

  a_B_P = a_B_Ao + np.cross(sum_alphas, r_Ao_P_B) + np.cross(sum_omegas, np.cross(sum_omegas, r_Ao_P_B))

  return r_Q_P_B, v_B_P, a_B_P