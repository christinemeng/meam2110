import numpy as np
import os
from meshcat import Visualizer
import meshcat.geometry
from urchin import URDF
import time
from . import Body, Joint, JointType, BodyVisual, PointKinematics, \
    RelativeRotationMatrix

class VisualOptions(object):
    '''
    Options for visualization 
    '''
    def __init__(self, visualize_frame = True, frame_scale = 0.3):
        '''
        Args:
          visualize_frame: If True, draws the coordinate frame for all bodies
          frame_scale: A factor to scale the visualized frame
        '''
        self.visualize_frame = visualize_frame
        self.frame_scale = frame_scale

class RigidBodySystem(object):
  '''
  The RigidBodySystem defines the relationship between the various `Body` and `Joint` elements, serving
  as the primary object of interest. Nearly all methods will require the RigidBodySystem to perform
  calculations. The prmary purpose of the RigidBodySystem is to define the kinematic tree that joins the
  bodys together. 

  RigidBodySystem also provides a number of utility methods for interrogating this tree structure.
  '''
  def __init__(self, visualize = True):
    '''
    Args:
      visualize: Determines whether or not to create a visualization. Defaults to True.
    '''
    self.bodies = dict() # (key, value) = (name, body)
    self.joints = dict() # (key, value) = (joint, index_into_position_vector)
    self.joint_to_state_index = [] # Not needed?
    self.state_to_joint_index = [] # Not needed?
    
    if visualize:
      self.visualizer = Visualizer()
    else:
      self.visualizer = None

    self.bodies['N'] = Body('N', 0, np.zeros((3,3)), np.zeros(3))
    #self.body_name_to_index(self.N.name) = 0
    self.force_torques = []

  def InertialFrameN(self):
    '''
    Return the inertal frame `N` (a `Body`)
    '''
    return self.bodies['N']
  
  def NumPositions(self):
    '''
    Return the number of position variables.
    Equivalent to the number of non-fixed joints
    '''
    return len(self.joint_to_state_index)

  def NumJoints(self):
    '''
    Return the number of joints, which includes fixed joints.
    '''
    return len(self.joints)
    
  def GetBodyByName(self, name):
    '''
    Get the `Body` object corresponding to the given name

    Args:
      name: The name (string)
    
    Returns:
      B: the body with that name, if it exists. None otherwise.
    '''
    return self.bodies[name]


  def AddJoint(self, parent, child, type, axis = None, r_Po_Jo = np.zeros(3), R_P_J = np.eye(3)):
    '''
    The primary function to construct systems. Creates and adds a joint to the system.

    Args:
      parent: The parent body
      child: The child body
      type: The JointType (fixed, rotation, translation)
      axis: The axis of rotation or translation, expressed in joint coordinates
      R_P_J: The rotation matrix from parent P to joint J \( ^P R^J \)
      r_Po_Jo: Expressed in P coordinates, \( ^{P_o} \\vec r^{J_o} \)

    Returns:
      joint: The created joint
    '''

    # Ensure that parent has already been added
    assert(parent.name in self.bodies), 'Parent body ' +  parent.name + ' has not yet been added to the system.'
    # Ensure that child has a unique name
    assert(child.name not in self.bodies), 'Child body ' + child.name + ' has already been added to the system.'

    # create and add joint
    joint = Joint(parent, child, type, axis, r_Po_Jo, R_P_J)


    # add body
    self.bodies[child.name] = child

    if type is not JointType.fixed:
      joint_index = self.NumJoints()
      state_index = self.NumPositions()
      self.joint_to_state_index.append(state_index)
      self.state_to_joint_index.append(joint_index)
    else:
      state_index = None
    self.joints[joint] = state_index

    return joint

  def AddForceTorque(self, force_torque):
    '''
    Add an applied `ForceTorque` to the system

    Args:
      force_torque: The ForceTorque object
    '''
    self.force_torques.append(force_torque)

  def ParseURDF(self, urdf, visual_options = VisualOptions()):
    '''
    Parse a URDF file, adding bodies and joints to the system. This is one of the primary ways to construct a RigidBodySystem

    <pre><code>system = RigidBodySystem
    system.ParseURDF('path_to_file.urdf')
    </code></pre>

    Args:
      urdf: (string) The path to the URDF file to parse
      visual_options: a `VisualOptions` objects (optional)
    '''
    robot = URDF.load(urdf)
    for joint in robot.joints:
      axis = joint.axis
      R_P_J = joint.origin[:3, :3]
      r_Po_Jo = joint.origin[:3, 3]
      if joint.joint_type == 'revolute':
        type = JointType.rotation
      elif joint.joint_type == 'prismatic':
        type = JointType.translation
      elif joint.joint_type == 'fixed':
        type = JointType.fixed

      # we merge the base link with 'N'
      if joint.parent == robot.base_link.name:
        parent = self.bodies['N']
      else:
        parent = self.bodies[joint.parent]

      # Find the child body
      found_child = False
      for link in robot.links:
        if link.name == joint.child:
          found_child = True

          # rotation from child frame to inertial frame
          R_C_I = link.inertial.origin[:3, :3]

          child_mass = link.inertial.mass

          child_inertia = R_C_I @ link.inertial.inertia @ R_C_I.T
          r_Bo_Bcm = link.inertial.origin[:3, 3]
          if self.visualizer is not None:
            child_vis = self.ParseVisual(urdf, link, visual_options)
          else:
            child_vis = None

          child = Body(link.name, child_mass, child_inertia, r_Bo_Bcm, child_vis)

      assert(found_child), 'Child \"' + link.name + '\" was not found in joint \"' + joint.name + '\"'
      self.AddJoint(parent, child, type, axis, r_Po_Jo, R_P_J)

  def Draw(self, q):
    '''
    Draw the current state of the system in the visualizer

    Args:
      q: A vector of all joint positions
    '''
    for B in self.bodies.values():
      r_No_Bo, _ , _ = PointKinematics(self, q, B, self.bodies['N'])
      R_N_B = RelativeRotationMatrix(self, q, B, self.bodies['N'])

      B.Draw(r_No_Bo, R_N_B)

  def Animate(self, t, q, *, qd = None, qdd = None, speed = 1.0, callbacks = []):
    '''
    Animates a trajectory in the visualizer. Default usage:

    <code>system.Animate(t, q)</code>

    Additional features can be visualized via the callback functions. Note that qd and qdd
    are not needed for basic visualization, but may be used by the callback functions.

    Args:
      t: A length `N` array of times
      q: A nxN ndarray of the joint positions, where q[:, i] is the position at time t[i]
      
      Optional arguments:
      
      qd: A nxN ndarray of the joint velocities, where qd[:, i] is the velocity at time t[i]
      qdd: A nxN ndarray of the joint accelerations, where qdd[:, i] is the acceleration at time t[i]
      speed: The playback speed. Defaults to 1.0
      callbacks: A list of callback visualization functions, where the callbacks have the format
         <code>callback(system, q, qd, qdd, t)</code>. Callbacks can be used, for example, to show angular velocity of bodies,
         positions of points, etc.
    '''
    i = 1
    t0 = time.time()

    while i > 0: # samples remain
      # configuration
      t_i = t[i]
      q_i = q[:, i]
      qd_i = qd[:, i] if qd is not None else None
      qdd_i = qdd[:, i] if qdd is not None else None

      self.Draw(q_i)
      for callback in callbacks:
        callback(self, q_i ,qd_i, qdd_i, t_i)

      time.sleep(.02)

      now = time.time() - t0
      i = np.argmax(t/speed > now)

  def ParseVisual(self, urdf_file, urdf_link, visual_options):
    '''
    A helper function to parse the <visual> tag of a <link> in a URDF
    '''
    name = urdf_link.name
    body_vis = None

    assert(len(urdf_link.visuals) <= 1), 'URDF links can only contain one visual element.'
    if len(urdf_link.visuals) == 1:
      visual = urdf_link.visuals[0]
      r_Bo_Vo = visual.origin[:3, 3]
      R_B_V = visual.origin[:3, :3]
      geometry = visual.geometry

      if visual.geometry.mesh is not None:
        mesh_file = visual.geometry.mesh.filename

        # build the full path for the mesh file
        full_file = urdf_file[:urdf_file.rfind('/') + 1] + mesh_file
        _, mesh_type = os.path.splitext(full_file)
        mesh_type = mesh_type[1:].lower()

        if mesh_type == 'dae':
          geom = meshcat.geometry.DaeMeshGeometry.from_file(full_file)
        elif mesh_type == 'stl':
          geom = meshcat.geometry.StlMeshGeometry.from_file(full_file)
        elif mesh_type == 'obj':
          geom = meshcat.geometry.ObjMeshGeometry.from_file(full_file)


      elif visual.geometry.box is not None:
        geom = meshcat.geometry.Box(visual.geometry.box.size)
      elif visual.geometry.cylinder is not None:
        cyl = visual.geometry.cylinder
        geom = meshcat.geometry.Cylinder(cyl.length, cyl.radius)
      elif visual.geometry.sphere is not None:
        geom = meshcat.geometry.Sphere(visual.geometry.sphere.radius)
      else:
        raise Exception('Did not find mesh, box, cylinder, or sphere geometry.')

      self.visualizer[name].set_object(geom)
      body_vis = BodyVisual(r_Bo_Vo, R_B_V, self.visualizer, name, \
          visual_options.visualize_frame, visual_options.frame_scale)
    return body_vis
