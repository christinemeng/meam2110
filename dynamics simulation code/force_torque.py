from abc import ABC, abstractmethod
import numpy as np
import sympy as sym
from . import PointKinematics, ChangeCoordinates, JointType, UnitVector

class ForceTorque(ABC):
  '''
  This class stores all geometric and physical information about applied
  forces and torques in a rigid body system. An object of this class
  represents:

   - a force ON point G (fixed on the child body C), FROM point H (fixed on the
     parent body P)

   - a torque ON the child body C FROM the parent body P

  By Newton's third law, an equal and opposite:

   - force on the parent P at point H

   - torque on the parent P will arise.

  Unless the parent is the inertial frame N, these reactions must also be
  factored into the dynamics of the system.

  This is an ABSTRACT class, meaning subclasses must be created to represent
  particular types of forces and torques (e.g. gravity, springs, etc.). Subclasses
  must implement the ComputeForceAndTorque(self, system, q, qdot) method.
  As an abstract class, you should never directly create a ForceTorque object.
  Instead, use the subclasses below.

  This class also stores references to two types of symbolic variables.
    specified_sym: values which must be specified by the user, as in the torque applied by a motor
    reaction_sym: values corresponding to reaction forces and torques,
      which must be solved for jointly with the equations of motion


  properties

    - name:   A string-valued name
    - C:      the "child" Body which is acted on by the force 
    - P:      the "parent" Body from which the force acts
    - r_Co_G: the vector from origin of C to G, in C coords
    - r_Po_H: the vector from origin of P to H, in P coords
    - specified_sym: a list of symbolic variables (see above)
    - reaction_sym: a list of symbolic variables (see above)
  '''
  def __init__(self, name, C, r_Co_G, P, r_Po_H, *, specified_sym = [], reaction_sym = []):
    self.name = name
    self.C = C
    self.r_Co_G = r_Co_G
    self.P = P
    self.r_Po_H = r_Po_H
    self.reaction_sym = reaction_sym
    self.specified_sym = specified_sym

  @abstractmethod
  def ComputeForceAndTorque(self, system, q, qdot):
    '''
    Computes the effect of a ForceTorque on its child body C. Note that Newton's
    Third Law implies an equal and opposite force and torque on P!

    Args:
      system: The RigidBodySystem object
      q: the vector of joint positions
      qdot: the vector of joint velocities

    Returns:
      F_C_N: The force applied on body C at G, in N coordinates, \\(\\vec F^C \\)
      T_C_N: The torque applied on body C, in N coordinates \\( \\vec T^C \\)
    '''

class GravityForce(ForceTorque):
  '''
  The force due to gravity (approximated as a constant vector field).

  - Parameterized by a single scalar constant "g" (specified as a *positive* value)
  - Has no associated reaction or specified symbolic variables

  The parent frame for gravity must be the inertial frame, and the force
  is applied at the bodies center of mass

  '''
  def __init__(self, system, B, g=9.81):
    '''
    Args:
      system: The `RigidBodySystem` object
      B: The body B that the force acts on
      g: The magnitude of the gravity force, defaults to 9.81 m/s^2
    '''
    super().__init__('gravity_' + B.name, B, B.r_Bo_Bcm, system.InertialFrameN(), np.zeros(3))
    self.g = g

  def ComputeForceAndTorque(self, system, q, qdot):
    '''
    See `ForceTorque.ComputeForceAndTorque` for documentation
    '''
    F_C_N = np.array([0, 0, -self.C.mass*self.g])
    T_C_N = np.zeros(3)
    return F_C_N, T_C_N

class SpringForce(ForceTorque):
  '''
  The force due to a linear spring attached between point G (fixed on body C)
  and point H (fixed on body P).

  Parameterized by two scalar constants (both are positive valued):
  
    - k, the spring constant
    - L0, natural rest length of the spring
  
  Has no associated reaction or specified symbolic variables
  '''
  def __init__(self, name, C, r_Co_G, P, r_Po_H, k, L0):
    '''
    Args:
      system: The `RigidBodySystem` object
      name: The string name of the spring force
      C: The child body
      r_Co_G: The position of the spring's attachment point on the child,
              relative to Co, in C's coordinates, \\( ^{C_o} \\vec r ^G \\)
      P: The parent body
      r_Po_H: The position of the spring's attachment point on the parent,
              relative to Po, in P's coordinates, \\( ^{P_o} \\vec r ^H \\)
      k: The spring constant
      L0: the rest length of the spring
    '''
    super().__init__(name, C, r_Co_G, P, r_Po_H)
    self.k = k
    self.L0 = L0

  def ComputeForceAndTorque(self, system, q, qdot):
    '''
    See `ForceTorque.ComputeForceAndTorque` for documentation
    '''
    if isinstance(q[0], sym.Expr):
      from sympy import sqrt
    else:
      from math import sqrt
    N = system.InertialFrameN()

    r_H_G_C, _, _ = PointKinematics(system, q, qdot = qdot, A = self.P, B = self.C, r_Ao_P = self.r_Po_H, r_Bo_Q = self.r_Co_G)
    r_H_G_N = ChangeCoordinates(system, q, r_H_G_C, self.C, N)
    length_r = sqrt(r_H_G_N[0] ** 2 + r_H_G_N[1] ** 2 + r_H_G_N[2] ** 2)
    unitVect_r = r_H_G_N / length_r
    F_C_N = (length_r - self.L0) * self.k * unitVect_r
    T_C_N = np.zeros(3)
    return F_C_N, T_C_N

class ConstantVelocityController(ForceTorque):
  '''
  An actuator attached to a joint that tries to keep that joint moving at constant velocity.
  For a translational joint, this means a linear actuator that applies a force.
  For a rorational joint, this is a rotational actautor that applies a torque.

  The applied force/torque value uses a proportional controller, with magnitude
  equal to K * (desired_velocity - velocity).

  If desired_velocity = 0, this is equivalent to a damper.
  '''
  def __init__(self, joint, desired_velocity, K):
    '''
    Args:
      joint: The `Joint` object
      desired_velocity: The commanded or desired velocity
      K: The gain magnitude
    '''
    super().__init__(joint.name, joint.child, np.zeros(3), joint.parent, joint.r_Po_Jo)
    self.joint = joint
    self.desired_velocity = desired_velocity
    self.K = K
  def ComputeForceAndTorque(self, system, q, qdot):
    '''
    See `ForceTorque.ComputeForceAndTorque` for documentation
    '''
    # Force/torque is axis * u, in the joint frame. Calculate in parent frame
    # and change coordinates to inertial frame
    velocity = qdot[system.joints[self.joint]]
    magnitude = self.K * (self.desired_velocity - velocity)
    if self.joint.type == JointType.rotation:
      F_C_N = np.zeros(3)

      T_C_P = self.joint.R_P_J @ self.joint.axis * magnitude
      T_C_N = ChangeCoordinates(system, q, T_C_P, self.P, system.InertialFrameN())
    elif self.joint.type == JointType.translation:
      F_C_P = self.joint.R_P_J @ self.joint.axis * magnitude
      F_C_N = ChangeCoordinates(system, q, F_C_P, self.P, system.InertialFrameN())
      T_C_N = np.zeros(3)

    return F_C_N, T_C_N

class JointActuator(ForceTorque):
  '''
  An actuator attached to a joint.  For a translational joint, this means a linear
  actuator that applies a force. For a rorational joint, this is a rotational actautor
  that applies a torque.

  The applied force/torque values come from the single `specified_sym` value.
  This symbolic variable is given the name `u_name` where name is the name of the ForceTorque object,
  which is the same as the name of the joint.
  '''
  def __init__(self, joint):
    '''
    Args:
      joint: The `Joint` object
    '''
    u_sym = sym.Symbol('u_' + joint.name)
    super().__init__(joint.name, joint.child, np.zeros(3), joint.parent,
        joint.r_Po_Jo, specified_sym = [u_sym])
    self.joint = joint

  def ComputeForceAndTorque(self, system, q, qdot):
    '''
    See `ForceTorque.ComputeForceAndTorque` for documentation
    '''
    # Force/torque is axis * u, in the joint frame. Calculate in parent frame
    # and change coordinates to inertial frame
    if self.joint.type == JointType.rotation:
      F_C_N = np.zeros(3)

      T_C_P = self.joint.R_P_J @ self.joint.axis * self.specified_sym[0]
      T_C_N = ChangeCoordinates(system, q, T_C_P, self.P, system.InertialFrameN())
    elif self.joint.type == JointType.translation:
      F_C_P = self.joint.R_P_J @ self.joint.axis * self.specified_sym[0]
      F_C_N = ChangeCoordinates(system, q, F_C_P, self.P, system.InertialFrameN())

      T_C_N = np.zeros(3)

    return F_C_N, T_C_N

class JointConstraint(ForceTorque):
  '''
  JointConstraint(joint)

  The constraint forces and torques associated with a joint. These are the forces
  and torques which eliminate the constrained degrees of freedom.

  For a joint which leaves 1 DOF, this means 5 constraint equations and associated
  forces and torques. Therefore, each JointConstraint object has 5-dimensional
  value for 'reaction_sym', which must be solved for alongside the equations of motion

  In the constructor, we create these 5 symbolic variables:

   - Forces: `RF_jointname_x`

   - Torques: `RT_jointname_`

   with the `R for reaction. The `x` at the end refers to an index [0,1,2].


  The implementation of ComputeForceAndTorque will define the exact meaning
  of these five variables.
  '''
  def __init__(self, joint):
    '''
    Args:
      joint: The `Joint` object
    '''
    self.joint = joint
    if self.joint.type == JointType.rotation:
      reaction_forces = np.array(sym.symbols('FR_' + joint.name + '_0:3'))
      reaction_torques = np.array(sym.symbols('TR_' + joint.name + '_0:2'))
    elif self.joint.type == JointType.translation:
      reaction_forces = np.array(sym.symbols('FR_' + joint.name + '_0:2'))
      reaction_torques = np.array(sym.symbols('TR_' + joint.name + '_0:3'))
    elif self.joint.type == JointType.fixed:
      reaction_forces = np.array(sym.symbols('FR_' + joint.name + '_0:3'))
      reaction_torques = np.array(sym.symbols('TR_' + joint.name + '_0:3'))
    u_sym = sym.Symbol('_' + joint.name)

    reaction_sym = np.hstack((reaction_forces, reaction_torques))
    super().__init__(joint.name, joint.child, np.zeros(3), joint.parent,
        joint.r_Po_Jo, reaction_sym = reaction_sym)


  def ComputeForceAndTorque(self, system, q, qdot):
    '''
    See `ForceTorque.ComputeForceAndTorque` for documentation
    '''
    if self.joint.type == JointType.rotation:
      F_C_J = np.array(self.reaction_sym[0:3])
      if np.array_equal(self.joint.axis, UnitVector.x):
        T_C_J = np.array([0, self.reaction_sym[3], self.reaction_sym[4]])
      elif np.array_equal(self.joint.axis, UnitVector.y):
        T_C_J = np.array([self.reaction_sym[3], 0, self.reaction_sym[4]])
      elif np.array_equal(self.joint.axis, UnitVector.z):
        T_C_J = np.array([self.reaction_sym[3], self.reaction_sym[4], 0])
    elif self.joint.type == JointType.translation:
      if np.array_equal(self.joint.axis, UnitVector.x):
        F_C_J = np.array([0, self.reaction_sym[0], self.reaction_sym[1]])
      elif np.array_equal(self.joint.axis, UnitVector.y):
        F_C_J = np.array([self.reaction_sym[0], 0, self.reaction_sym[1]])
      elif np.array_equal(self.joint.axis, UnitVector.z):
        F_C_J = np.array([self.reaction_sym[0], self.reaction_sym[1], 0])
      T_C_J = np.array(self.reaction_sym[2:5])
    elif self.joint.type == JointType.fixed:
      F_C_J = self.reaction_sym[0:3]
      T_C_J = self.reaction_sym[3:6]

    F_C_P = self.joint.R_P_J @ F_C_J
    T_C_P = self.joint.R_P_J @ T_C_J

    F_C_N = ChangeCoordinates(system, q, F_C_P, self.P, system.InertialFrameN())
    T_C_N = ChangeCoordinates(system, q, T_C_P, self.P, system.InertialFrameN())

    return F_C_N, T_C_N


def AddJointConstraints(system):
  '''
  This function adds ForceTorque objects to the system which model the
  constraint forces arising at all Joint's in the system, where
  each ForceTorque has type JointConstraint and params which define the
  reaction forces that arise at that joint in accordance with its type.
  Note that reaction components should only be included in components
  which are constrained by the joint, not those that are free!

  Args:
    system: The RigidBodySystem object
  '''

  # iterate through all the joints in the system
  for joint in system.joints.keys():
      constraint_force = JointConstraint(joint)
      system.AddForceTorque(constraint_force)

def AggregateReactionVariables(system):
  '''
  Collects symbolic reaction force and torque variables 
  from all `ForceTorque`s in the system. See the definition of `ForceTorque`

  Args:
    system: The RigidBodySystem object

  Returns:
    reaction_vars: A column vector of all symbolic variables for reaction forces/torques
  '''
  reaction_vars = np.array([])
  for force_torque in system.force_torques:
    reaction_vars = np.hstack((reaction_vars, force_torque.reaction_sym))
  return reaction_vars

def AggregateSpecifiedVariables(system):
  '''
  Collects symbolic specified force and torque variables 
  from all `ForceTorque`s in the system. See the definition of `ForceTorque`

  Args:
    system: The RigidBodySystem object

  Returns:
    specified_vars: A column vector of all symbolic variables for specified forces/torques
  '''
  specified_vars = np.array([])
  for force_torque in system.force_torques:
    specified_vars = np.hstack((specified_vars, force_torque.specified_sym))
  return specified_vars

def ComputeAppliedForcesAndMoments(system, q, qdot, B, A = None):
  '''
  Compute the total force and total moment (about the center of mass) on a specific body,
  due to all applied forces and moments, expressed in A coordinates

  Args:
    system: The RigidBodySystem object
    q: The vector of joint positions
    qdot: The vector of joint velocities
    B: The body B whose total applied forces and moments are being computed 
    A: The frame in which the results are expressed. Defaults to N if not given as an input

  Returns:
    F_B_A: the total applied force on body B, in A coords
    M_Bcm_A: the total net moment on body B about its center of mass,
              expressed in A coords, due to the combination of:

       - applied pure torques

       - the moment of applied forces about the center of mass, which arise due
          to the points of application of said forces
  '''

  # ... then convert to A
  F_B_A = np.zeros(3)
  M_Bcm_A = np.zeros(3)


  if A is None:
    A = system.InertialFrameN()

  for force_torque in system.force_torques:
    # Call the point on B where the force is applied "E"
    # E will either by G on the child body or H on the parent body
    # See ForceTorque class definition above

    # Suggested approach:
    # Check if the ForceTorque acts on the body (if the ForceTorque's parent or child is B)
    # If it does, add in the effect of the ForceTorque

    N = system.InertialFrameN()
    new_F_N, new_T_N = force_torque.ComputeForceAndTorque(system=system, q=q, qdot=qdot)
    new_F_A = ChangeCoordinates(system, q, new_F_N, N, A)
    new_T_A = ChangeCoordinates(system, q, new_T_N, N, A)


    # YOUR CODE GOES BELOW!
    if B == force_torque.C:
      # B = C, child body in ForceTorque.m
      # Force is applied at point G=E
      # None # delete this line when you're ready to code here
      r_Bcm_E_B = force_torque.r_Co_G-force_torque.C.r_Bo_Bcm
      r_Bcm_E_A = ChangeCoordinates(system, q, r_Bcm_E_B, force_torque.C, A)
      F_B_A = F_B_A + new_F_A
      M_Bcm_A = M_Bcm_A + new_T_A + np.cross(r_Bcm_E_A, new_F_A)
    elif B == force_torque.P:
      r_Bcm_E_B = force_torque.r_Po_H-force_torque.P.r_Bo_Bcm
      r_Bcm_E_A = ChangeCoordinates(system, q, r_Bcm_E_B, force_torque.P, A)
      F_B_A = F_B_A - new_F_A
      M_Bcm_A = M_Bcm_A - new_T_A - np.cross(r_Bcm_E_A, new_F_A)
      # B = P, parent body in ForceTorque.m
      # Force is applied at point H=E
      # None # delete this line when you're ready to code here
    else:
      # this force does not act on or from body B
      continue

    # add to get net force and moment about center of mass, accounting
    # for the point of application F of the force
    # F_B_A = F_B_A
    # M_Bcm_A = M_Bcm_A
  return F_B_A, M_Bcm_A



def AddGravityToSystem(system, g=9.81):
  '''
  Add GravityForce objects for all bodies (that have mass) to the system

  Args:
    system: The `RigidBodySystem` object
    g: The magnitude of the gravity force, defaults to 9.81 m/s^2
  '''
  for B in system.bodies.values():
    if B.mass > 0:
      system.AddForceTorque(GravityForce(system, B, g))