import numpy as np
from scipy.spatial.transform import Rotation


class Transform:
    def __init__(self, matrix = np.eye(4)):
        self.matrix = np.array(matrix)
    
    def rotation(self):
        return self.matrix[:3,:3]
    
    def rotation_gl(self):
        return [0,0]
    
    def translation(self):
        return self.matrix[:3,3]

    def position(self):
        return self.matrix[:3,3]
    
    def position4(self):
        return self.matrix[:,3]
    
    def apply(self, points):
        points = np.array(points)
        if points.ndim == 1: points = points[None]
        points = np.concatenate([points, np.ones([len(points),1])], 1)
        return np.squeeze((points @ self.matrix.T)[:,:3])
    
    def inverse(self):
        M = self.matrix.copy()
        M[:3,:3] = self.rotation().T
        M[:3,3] = - self.rotation().T @ self.translation()
        return Transform(M)
    
    def local(self, point):
        return point - self.translation()
    
    def mul(self, other):
        return Transform(self.matrix @ other.matrix)
    
    def __repr__(self):
        return np.array2string(self.matrix, precision = 4, suppress_small=True)
    
    @staticmethod
    def from_euler(angles):
        mat = np.eye(4)
        mat[:3,:3] = Rotation.from_euler('yzx', np.deg2rad(angles)).as_matrix()
        return Transform(mat)

    @staticmethod
    def from_vec(vec):
        mat = np.eye(4)
        mat[:3,3] = vec
        return Transform(mat)

    @staticmethod
    def from_rot_pos(angles, vec):
        return Transform.from_vec(vec).mul(Transform.from_euler(angles))