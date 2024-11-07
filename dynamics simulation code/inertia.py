import numpy as np
from. import PointKinematics, RelativeRotationMatrix

# seems like all this is already solved?

def SystemMass(system):
  '''
  Compute the total mass of the system m_S

  Args:
    system: The RigidBodySystem object

  Returns:
    m_S: The total mass of the system
  '''

  m_S = 0
  for body in system.bodies.values():
    m_S += body.mass
  return m_S

def SystemCOM(system, q, B):
  '''
  Compute the center of mass of the system from point Bo, expressed in B's coordinates

  Args:
    system: The RigidBodySystem oject
    q: The current system configuration
    B: Frame B

  Returns:
    r_Bo_Scm: Center of mass of the system from point Bo, \(^{B_o}\\vec r^{S_{cm}} \) expressed in  B's coordinates
  '''
  r_Bo_Scm = np.zeros(3)
  m_S = SystemMass(system)

  for body in system.bodies.values():
    r_Bo_Bodycm, _, _ = PointKinematics(system, q, body, B, r_Ao_P = body.r_Bo_Bcm)
    r_Bo_Scm = r_Bo_Scm + body.mass / m_S * r_Bo_Bodycm

  return r_Bo_Scm

def ChangeInertiaOrigin(system, B, r_Bcm_P):
  '''
  I_B_P = ChangeInertiaOrigin(system, B, r_Bcm_P)

  Changes origin of inertia of body B from Bcm to arbitrary point P

  system: A RigidBodySystem object
  B: The body described by the inertia tensor
  r_Bcm_P" Vector from Bcm to point P, in B coordinates
  I_B_P: Inertia of B about point P, in B coordinates
  '''

  I_B_P = B.I_B_Bcm + B.mass * (r_Bcm_P.dot(r_Bcm_P) * np.eye(3) - np.outer(r_Bcm_P, r_Bcm_P.T))

  return I_B_P


def ChangeInertiaCoordinates(system, q, I_S_P_A, A, B):
  '''
  I_S_P_B = ChangeInertiaCoordinates(system, q, I_S_P_A, A, B)
  Changes the given inertia matrix from A's coordinates to B's coordinates.

  system: The RigidBodySystem object
  q: The current system configuration
  I_S_P_A: Inertia of S around point P in A's coordinates
  A: The body whose coordinates are used to express I_S_P_A
  B: The body whose coordinates will be used to express I_S_P_B

  I_S_P_B: The inertia of S around P in B coordinates
  '''

  R_B_A = RelativeRotationMatrix(system, q, A, B)

  I_S_P_B = R_B_A @ I_S_P_A @ R_B_A.T

  return I_S_P_B