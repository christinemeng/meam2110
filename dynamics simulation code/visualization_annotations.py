import numpy as np
from scipy.spatial.transform import Rotation
from meshcat.geometry import Cylinder, Sphere, MeshLambertMaterial
from . import PointKinematics, BodyAngVelAndAccel, CreateTransform

def DrawAngularVelocityVector(system, B, q, qdot, thickness, scale):
  '''
  Draw angular velocity of a body w_N_B as a semi-transparent black cylinder, located at the body origin

  Args:
    system: The rigid body system
    B: The body B
    q: positions
    qdot: velocities
    thickness: thickness of the cylinder
    scale: scaling factor for the radius
  '''
  r_No_Bo, _, _ = PointKinematics(system, q, B, system.InertialFrameN())
  w_N_B,_ = BodyAngVelAndAccel(system, q, qdot, B, system.InertialFrameN())

  cyl = Cylinder(thickness, scale*np.linalg.norm(w_N_B))
  material = MeshLambertMaterial(color=0x000000, opacity=.2)
  system.visualizer[B.name + '_angvel'].set_object(cyl, material)

  if np.linalg.norm(w_N_B) > 1e-3:
    R_N_AV = Rotation.align_vectors([w_N_B], [np.array([0, 1, 0])])[0].as_matrix()

  else:
    R_N_AV = np.eye(3)
  system.visualizer[B.name + '_angvel'].set_transform(CreateTransform(r_No_Bo, R_N_AV))


def AngularVelocityAnnotation(system, B, thickness, scale):
  '''
  Returns the annotation, for use in RigidBodySystem.Animate, that draws
  the angular velocity vector of the given body

  Args:
    system: The RigidBodySystem
    B: The body to draw angular velocity for
    thickness: The thickness of the velocity vector
    scale: Scales the drawn vector by this factor
  '''
  annotation = lambda sys, q, qdot, qddot, t : DrawAngularVelocityVector(sys, B, q, qdot, thickness, scale)
  return annotation


def DrawArrow(vis, name, origin, vector, radius, color, opacity):
  '''
  Draw an "arrow" using a cylinder and a sphere for the arrowhead

  Args:
    vis: the visualizer object
    name: a name to use for the arrow
    origin: the origin of the arrow
    vector: the arrow to draw, starting at the origin point
    radius: the radius to use for the arrow
    color: arrow color
    opacity: arrow opacity
  '''
  cyl_len = np.linalg.norm(vector)
  cyl = Cylinder(cyl_len, radius)
  # Set the color to be mostly transparent red.
  material = MeshLambertMaterial(color=color, opacity=opacity)
  vis[name + '_cyl'].set_object(cyl, material)

  # Set the location and orientation.
  if cyl_len > 1e-3:
      R_N_Arrow = Rotation.align_vectors([vector], [np.array([0, 1, 0])])[0].as_matrix()
  else:
      R_N_Arrow = np.eye(3)

  # Set the location so the base of the cylinder is at the origin
  r_No_Vo = origin + vector / 2
  vis[name + '_cyl'].set_transform(CreateTransform(r_No_Vo, R_N_Arrow))

  # Add a sphere at the end to indicate the direction.
  r_No_So = origin + vector

  if cyl_len > 1e-3:
    sphere = Sphere(3*radius)
  else:
    sphere = Sphere(1e-3)
  vis[name + '_sphere'].set_object(sphere, material)
  vis[name + '_sphere'].set_transform(CreateTransform(r_No_So, R_N_Arrow))

def DrawForceTorque(sys, force_torque, radius, scale, q, qdot):
  '''
  Draw arrows for the force and torque corresponding to a ForceTorque object
  Draws a red arrow for the force, starting at the point of action for the force
  Draws a blue arrow for the torque, starting from same location
  Arrows scale in magnitude with the force and torque

  Args:
    sys : RigidBodySystem object
    force_torque: The ForceTorque object
    radius: The radius for the body of the arrow
    scale: A scaling factor applied to the arrow size
    q: System positions
    qdot: System velocities
  '''
  # Compute the force and torque vectors.  The annotation will draw
  # each in red and blue, respectively.
  F_C_N, T_C_N = force_torque.ComputeForceAndTorque(sys, q, qdot)

  # Calculate the location of the force in world frame.
  child_body = force_torque.C
  N = sys.InertialFrameN()

  # force is applied at point G
  r_No_G, _, _ = PointKinematics(sys, q, child_body, N, r_Ao_P = force_torque.r_Co_G)

  # Call the force torque vector draw function.
    # 1) Draw force vector first.
  # Make a narrow cylinder with length that scales with force
  # magnitude and whose base begins at the application location.
  DrawArrow(sys.visualizer, force_torque.name + '_force', r_No_G, scale * F_C_N, radius, 0xff0000, 0.2)

  # 2) Draw torque vector second.
  # Make a narrow cylinder with length that scales with torque
  # magnitude and whose base begins at the application location.
  DrawArrow(sys.visualizer, force_torque.name + '_torque', r_No_G, scale * T_C_N, radius, 0x0000ff, 0.2)

def ForceTorqueAnnotation(system, force_torque, radius, scale):
  '''
  Returns the annotation, for use in RigidBodySystem.Animate, that draws
  the angular velocity vector of the given body

  Args:
    system: The rigid body system
    force_torque: The ForceTorque object
    radius: The radius for the body of the arrow
    scale: A scaling factor applied to the arrow size
  '''
  annotation = lambda sys, q, qdot, qddot, t : DrawForceTorque(sys, force_torque, radius, scale, q, qdot)
  return annotation
  