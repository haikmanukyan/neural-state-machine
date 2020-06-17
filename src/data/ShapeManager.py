import numpy as np

class ShapeManager(object):
    def get_vector(self, data, name1, name2 = 'foo'):
        indices = []
        for i,x in enumerate(data):
            if name1 in x or name2 in x:
                indices.append(i)
        return indices

    def get_xz(self,data,name1,name2 = ''):
        indices = []
        for i,x in enumerate(data):
            if name1 in x and name2 in x and x[-1] in ['X','Z']:
                indices.append(i)
        return indices
    
    def get_xyz(self, data, name1, name2 = ''):
        indices = []
        for i,x in enumerate(data):
            if name1 in x and name2 in x and x[-1] in ['X', 'Y', 'Z']:
                indices.append(i)
        return indices

    def get_enum(self, data, name1, name2 = ''):
        indices = []
        for i,x in enumerate(data):
            if name1 in x and name2 in x and x[-1] not in ['X','Y','Z']:
                indices.append(i)
        return indices

    def get(self, data, name):
        return data[:,getattr(self,name)]

    def get_bounds(self, data, name):
        bounds = getattr(self, name + '_bounds')
        if len(data.shape) == 1:
            return data[bounds[0]:bounds[1]]
        else:
            return data[:,bounds[0]:bounds[1]]


class InputShapeManager(ShapeManager):
    def __init__(self):
        with open('./data/InputLabels.txt') as f:
            self.file = f.read().split('\n')

        self.indices = {}
        self.bones = self.get_vector(self.file, 'Bone')
        
        self.trajectory = self.get_xz(self.file, 'Trajectory')
        self.trajectory_position = self.get_xz(self.file, 'Trajectory', 'Position')
        self.trajectory_direction = self.get_xz(self.file, 'Trajectory', 'Direction')
        self.trajectory_actions = self.get_enum(self.file, 'Trajectory')

        self.frame = self.get_vector(self.file, 'Bone', 'Trajectory')

        self.goal = self.get_vector(self.file, 'Goal', 'Action')
        self.goal_position = self.get_xyz(self.file, 'Goal','Position')
        self.goal_direction = self.get_xyz(self.file, 'Goal','Direction')
        self.goal_actions = self.get_enum(self.file, 'Action')

        self.environment = self.get_vector(self.file, 'Environment')
        self.interaction = self.get_vector(self.file, 'Interaction')
        self.gating = self.get_vector(self.file, 'Gating')

        self.frame_bounds = [np.min(self.frame), np.max(self.frame)+1]
        self.goal_bounds = [np.min(self.goal), np.max(self.goal)+1]
        self.environment_bounds = [np.min(self.environment), np.max(self.environment)+1]
        self.interaction_bounds = [np.min(self.interaction), np.max(self.interaction)+1]
        self.gating_bounds = [np.min(self.gating), np.max(self.gating)+1]

class OutputShapeManager(ShapeManager):
    def __init__(self):
        with open('./data/OutputLabels.txt') as f:
            self.file = f.read().split('\n')
        
        self.indices = {}
        self.bones = self.get_vector(self.file, 'Bone')
        
        self.trajectory = self.get_xz(self.file, 'Trajectory')
        self.trajectory_position = self.get_xz(self.file, 'Trajectory', 'Position')
        self.trajectory_direction = self.get_xz(self.file, 'Trajectory', 'Direction')
        self.trajectory_actions = self.get_enum(self.file, 'Trajectory')

        self.frame = self.get_vector(self.file, 'Bone', 'Trajectory')

        self.goal = self.get_vector(self.file, 'Goal', 'Action')
        self.goal_position = self.get_xyz(self.file, 'Goal','Position')
        self.goal_direction = self.get_xyz(self.file, 'Goal','Direction')
        self.goal_actions = self.get_enum(self.file, 'Action')

        self.phase = self.get_vector(self.file, 'PhaseUpdate')

        self.frame_bounds = [np.min(self.frame), np.max(self.frame)+1]
        self.goal_bounds = [np.min(self.goal), np.max(self.goal)+1]
        self.phase_bounds = [np.min(self.phase), np.max(self.phase)+1]

class Data(object):
    def __init__(self, data, shape_manager):
        if shape_manager == "input":
            self.shape = InputShapeManager()
        elif shape_manager == "output":
            self.shape = OutputShapeManager()
        else:
            self.shape = shape_manager
            
        self.data = data
        self.keys = super().__getattribute__("__dict__").keys()

    def __getattribute__(self, name):
        if name == "keys" or name in self.keys:
            return super().__getattribute__(name)
        else:
            return self.shape.get_bounds(self.data, name)



if __name__ == "__main__":
    data = np.random.random((2000,10000))
    x = Data(data, OutputShapeManager())
    print (x.goal)
