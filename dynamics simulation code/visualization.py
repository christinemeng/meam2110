import numpy as np
from meshcat.geometry import triad

class BodyVisual(object):
  '''
  Visual information for a body.
  This class stores information necessary to visualize a body B using Meshcat.
  Uses a frame V to represent the visualization frame, where V is fixed in B.
  '''
  def __init__(self, r_Bo_Vo, R_B_V, vis, name,  visualize_frame = False, frame_scale = 1):
    self.r_Bo_Vo = r_Bo_Vo
    self.R_B_V = R_B_V
    self.vis = vis
    self.name = name
    self.visualize_frame = visualize_frame

    if self.visualize_frame:
      # Create a triad object
      body_triad = triad(frame_scale)
      self.vis[self.name + '_triad'].set_object(body_triad)

  def Draw(self, r_No_Bo, R_N_B):
    '''
    Draw the body given the position and orientation

    Args:
      r_No_Bo: The position of Bo w.r.t. the inertial origin No, \( ^{N_o} \\vec r^{B_o} \)
      R_N_B: The rotation matrix from inetial frame N to body B \( ^N R^B \)
    '''
    r_No_Vo = r_No_Bo + R_N_B @ self.r_Bo_Vo
    R_N_V = R_N_B @ self.R_B_V


    T_N_V = CreateTransform(r_No_Vo, R_N_V)
    self.vis[self.name].set_transform(T_N_V)

    if self.visualize_frame:
      # Draw triad at body origin, not visual origin
      T_N_B = CreateTransform(r_No_Bo, R_N_B)
      self.vis[self.name + '_triad'].set_transform(T_N_B)

def CreateTransform(r, R):
  '''
  Arrange rotation and translation into 4x4 transform (not taught in MEAM 2110)
  $$
  T = \\begin{bmatrix} & R & & r \\\ 0 & 0 & 0 & 1 \end{bmatrix}
  $$
  '''
  return np.vstack((np.hstack((R, r.reshape((3,1)))), np.array((0, 0, 0, 1))))

