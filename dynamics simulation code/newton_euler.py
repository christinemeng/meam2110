import numpy as np
import sympy as sym
from . import PointKinematics, ComputeAppliedForcesAndMoments, ChangeCoordinates, \
    BodyAngVelAndAccel, AggregateReactionVariables, SimplifyAndRound, RoundSmallCoefficients

def NewtonEulerBodyEquations(system, q, qdot, qddot_sym, B):
  '''  
  Compute the Newton-Euler equations for a single rigid body B

  Args:
    system: The RigidBodySysteml object
    q: The vector of joint positions
    qdot: The vector of joint velocities
    qddot_sym: A **symbolic** vector of joint accelerations
    B: The Body object
  
  Returns:
    eqn: A 6x1 vector of the Newton-Euler equations, written as:
    $$ \\begin{bmatrix}
    F - ma \\\\ M - I \\alpha - \\omega \\times (I \\omega)
    \\end{bmatrix}$$
    For appropriate \\(F, a, M, I, \\omega, and \\alpha \\)
  '''
  # YOUR CODE GOES HERE
  N = system.InertialFrameN()
  F_N, M_N = ComputeAppliedForcesAndMoments(system = system, q = q, qdot = qdot, B = B, A = N)
  m = B.mass
  _, _, a = PointKinematics(system = system, q = q, A = B, B = N, qdot = qdot, qddot = qddot_sym, r_Ao_P = B.r_Bo_Bcm)
  eqn_linear = F_N - m * a

  omega_N, alpha_N = BodyAngVelAndAccel(system = system, q = q, qdot = qdot, A = B, B = N, qddot = qddot_sym)
  omega_B = ChangeCoordinates(system, q, omega_N, N, B)
  alpha_B = ChangeCoordinates(system, q, alpha_N, N, B)
  M_B = ChangeCoordinates(system, q, M_N, N, B)
  I_B = B.I_B_Bcm
  eqn_angular = M_B - I_B @ alpha_B - np.cross(omega_B, I_B @ omega_B)

  eqn = np.hstack((eqn_linear, eqn_angular))
  return eqn


def NewtonEulerBodySymbolicEquations(system, B, simplify=True):
  '''
  A helper function to create a function (python labmda) f(t,x) where
  d/dt x = f(t,x) for a single body.

  This function works by

  1. Creating appropriate symbolic variables
  
  2. Calling NewtonEulerBodyEquations
  
  3. Solving the resulting system of equations for accelerations (qddot) and reaction forces

  Since this function only acts on a single body, it should only be called in simple situations
  where the entire system is a single body.

  This function has no dynamics/kinematics content, but helps with the use of python symbolic
  variables.

  Args:
    system: The RigidBodySystem object
    B: The body of interest
    simplify: Determines whether symbolic simplificaiton is used. Simplification can be slow, so this
            may need to be disabled for more complex examples

  Returns:
    xdot_fun: A lambda function `f` expecting two inputs, a scalar t and vector x. Computes d/dt x = f(t,x)
  '''
  n_q = system.joint_to_state_index[-1]
  q_sym = np.array(sym.symbols('q' + '_0:' + str(n_q + 1)))
  qdot_sym = np.array(sym.symbols('qdot' + '_0:' + str(n_q + 1)))
  qddot_sym = np.array(sym.symbols('qddot' + '_0:' + str(n_q + 1)))

  eom = NewtonEulerBodyEquations(system, q_sym, qdot_sym, qddot_sym, B)
  if simplify:
    eom = SimplifyAndRound(eom)
  else:
    eom = RoundSmallCoefficients(eom)
  reaction_vars = AggregateReactionVariables(system)
  solution = sym.solve(eom, np.hstack((qddot_sym, reaction_vars)).tolist(), dict=False)

  xdot_fun = EquationsOfMotionToFunction(system, q_sym, qdot_sym, qddot_sym, eom)
  return xdot_fun


def NewtonEulerSystemEquations(system, q, qdot, qddot_sym):
  '''
  Compute the equations of motion for a system of rigid bodies.

  Args:
    system: The RigidBodySystem being modeled
    q: The vector of joint positions
    qdot: The vector of joint velocities
    qddot_sym: A **symbolic** vector of joint accelerations

  Returns:
    eqns: A 6m x 1 vector of equations, eqns = [eqns_1, eqns_2, ..., eqns_m] for each body
          in the system. See NewtonEulerBodyEquations for more detail on structure
  '''

  eqns = np.array([])

  # Comptue N-E equations for each body, sipping N
  for B in system.bodies.values():
    if B is not system.InertialFrameN():
      eqn = NewtonEulerBodyEquations(system, q, qdot, qddot_sym, B)
      eqns = np.hstack((eqns, eqn))
  return eqns

def NewtonEulerSystemSymbolicEquations(system, simplify=True):
  '''
  A helper function to create a function (python labmda) f(t,x) where
  d/dt x = f(t,x) for the entire system

  This function works by
  
  1. Creating appropriate symbolic variables

  2. Calling NewtonEulerSystemEquations

  3. Solving the resulting system of equations for accelerations (qddot) and reaction forces

  This function has no dynamics/kinematics content, but helps with the use of python symbolic
  variables.

  Args:
    system: The RigidBodySystem object
    simplify: Determines whether symbolic simplificaiton is used. Simplification can be slow, so this
              may need to be disabled for more complex examples

  Returns:
    xdot_fun: A lambda function `f` expecting two inputs, a scalar t and vector x. Computes d/dt x = f(t,x)
  '''
  n_q = system.joint_to_state_index[-1]
  q_sym = np.array(sym.symbols('q' + '_0:' + str(n_q + 1)))
  qdot_sym = np.array(sym.symbols('qdot' + '_0:' + str(n_q + 1)))
  qddot_sym = np.array(sym.symbols('qddot' + '_0:' + str(n_q + 1)))

  eom = NewtonEulerSystemEquations(system, q_sym, qdot_sym, qddot_sym)
  if simplify:
    eom = SimplifyAndRound(eom)
  else:
    eom = RoundSmallCoefficients(eom)
  reaction_vars = AggregateReactionVariables(system)

  xdot_fun = EquationsOfMotionToFunction(system, q_sym, qdot_sym, qddot_sym, eom)
  return xdot_fun

def EquationsOfMotionToFunction(system, q_sym, qdot_sym, qddot_sym, eom):
  '''
  A helper function which converts a symbolic expression for the equations of motion into a lambda function

  Args:
    system: The RigidBodySystem object
    q_sym: An array of `sympy.Syms` for the position variables q
    qdot_sym: An array of `sympy.Syms` for the velocity variables qdot
    qddot_sym: An array of `sympy.Syms` for the acceleration variables qddot
    eom: An array of the equations of motion

  Returns:
    xdot_fun: A lambda function xdot = f(t,x)
  '''

  # remove equations that are exactly zero
  eom = sym.ImmutableDenseNDimArray([eq for eq in eom if eq != 0])

  reaction_vars = AggregateReactionVariables(system)
  qddot_reaction = qddot_sym.tolist() + reaction_vars.tolist()

  # Exploits the fact that equations of motion are linear in qddot and reaction
  # forces. Find A, b such that A*[qddot; reaction_forces] = b
  A,b = sym.linear_eq_to_matrix(eom, qddot_reaction)

  # Turn A,b into functions, then solve and stack with velocities for xdot_fun
  nq = q_sym.shape[0]
  A_lambda = sym.lambdify(np.hstack((q_sym, qdot_sym)), A, modules='numpy')
  b_lambda = sym.lambdify(np.hstack((q_sym, qdot_sym)), b, modules='numpy')
  xdot_fun = lambda t, x : np.hstack((x[nq:], np.linalg.solve(A_lambda(*x), b_lambda(*x))[0:nq,0]))

  return xdot_fun
