import numpy as np

class ShapeManager(object):
    def get_vector(self, data, name1, name2 = 'foo'):
        indices = []
        for i,x in enumerate(data):
            if name1 in x or name2 in x:
                indices.append(i)
        return np.array(indices)

    def get_xz(self,data,name1,name2 = ''):
        indices = []
        for i,x in enumerate(data):
            if name1 in x and name2 in x and x[-1] in ['X','Z']:
                indices.append(i)
        return np.array(indices).reshape(-1,2)
    
    def get_xyz(self, data, name1, name2 = ''):
        indices = []
        for i,x in enumerate(data):
            if name1 in x and name2 in x and x[-1] in ['X', 'Y', 'Z']:
                indices.append(i)
        return np.array(indices).reshape(-1,3)

    def get_enum(self, data, name1, name2 = ''):
        indices = []
        names = []
        for i,x in enumerate(data):
            if name1 in x and name2 in x and x[-1] not in ['X','Y','Z']:
                indices.append(i)
                names.append(x.split('-')[-1])
        return np.array(indices).reshape(-1,len(set(names))), np.array(names[:len(set(names))])

    def get(self, data, name):
        if "labels" in name:
            return getattr(self, name)[data[...,getattr(self,name[:-len("_labels")])].argmax(-1)]
        else:
            return data[...,getattr(self,name)]

        # if len(data.shape) == 1:
        #     return data[getattr(self,name)]
        # else:
        #     return data[:,getattr(self,name)]

    def set(self, data, name, value):
        if len(data.shape) == 1:
            data[getattr(self,name)] = value
        else:
            data[:,getattr(self,name)] = value

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

        self.dims = len(self.file)
        self.indices = {}
        self.bones = self.get_vector(self.file, 'Bone')
        self.bone_position = self.get_xyz(self.file, 'Bone', 'Position')
        
        self.trajectory = self.get_vector(self.file, 'Trajectory')
        self.trajectory_position = self.get_xz(self.file, 'Trajectory', 'Position')
        self.trajectory_direction = self.get_xz(self.file, 'Trajectory', 'Direction')
        self.trajectory_action, self.trajectory_action_labels = self.get_enum(self.file, 'Trajectory')

        self.frame = self.get_vector(self.file, 'Bone', 'Trajectory')

        self.goal = self.get_vector(self.file, 'Goal', 'Action')
        self.goal_position = self.get_xyz(self.file, 'Goal','Position')
        self.goal_direction = self.get_xyz(self.file, 'Goal','Direction')
        self.goal_action, self.goal_action_labels = self.get_enum(self.file, 'Action')

        self.environment = self.get_vector(self.file, 'Environment')
        self.interaction = self.get_vector(self.file, 'Interaction')
        self.interaction_position = self.get_xyz(self.file, 'InteractionPosition')
        self.interaction_occupancy = self.get_vector(self.file, 'InteractionOccupancy')
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
        
        self.dims = len(self.file)
        self.indices = {}
        self.bones = self.get_vector(self.file, 'Bone')
        self.bone_positions = self.get_vector(self.file, 'Bone', 'Position')
        
        self.trajectory = self.get_vector(self.file, 'Trajectory')
        self.trajectory_position = self.get_xz(self.file, 'Trajectory', 'Position')
        self.trajectory_direction = self.get_xz(self.file, 'Trajectory', 'Direction')
        self.trajectory_action, self.trajectory_action_labels = self.get_enum(self.file, 'Trajectory')

        self.frame = self.get_vector(self.file, 'Bone', 'Trajectory')

        self.goal = self.get_vector(self.file, 'Goal', 'Action')
        self.goal_position = self.get_xyz(self.file, 'Goal','Position')
        self.goal_direction = self.get_xyz(self.file, 'Goal','Direction')
        self.goal_action, self.goal_action_labels = self.get_enum(self.file, 'Action')

        self.phase = self.get_vector(self.file, 'PhaseUpdate')

        self.frame_bounds = [np.min(self.frame), np.max(self.frame)+1]
        self.goal_bounds = [np.min(self.goal), np.max(self.goal)+1]
        self.phase_bounds = [np.min(self.phase), np.max(self.phase)+1]

class DataBase(object):
    def __init__(self, data, norm, shape_manager, sequences = None):
        if sequences is None:
            self.sequences = np.zeros(len(data), np.int32)
        else:
            if type(sequences) is str:
                sequences = np.loadtxt(sequences)
            
            self.sequences = sequences

        self.clips = [np.where(sequences == i)[0] for i in np.unique(sequences)]
        self.norm = norm[:,:data.shape[-1]]
        self.mean, self.std = self.norm
        self.dims = min(len(self.mean), data.shape[-1])

        if shape_manager == "input":
            self.shape = InputShapeManager()
        elif shape_manager == "output":
            self.shape = OutputShapeManager()
        else:
            self.shape = shape_manager

        self.data = data

    def normed(self):
        return (self.data[...,:self.dims] - self.mean) / (1e-5 + self.std)
    
    def get_normed(self):
        normed_data = (self.data[...,:self.dims] - self.mean) / (1e-5 + self.std)
        return Data(normed_data, self.norm, self.shape, self.sequences)
    
    def normalize(self):
        self.data[...,:self.dims] -= self.mean[...,:self.dims]
        self.data[...,:self.dims] /= (1e-5 + self.std[...,:self.dims])
        return self

    def unnormalize(self):
        self.data[...,:self.dims] *= (1e-5 + self.std[...,:self.dims])
        self.data[...,:self.dims] += self.mean[...,:self.dims]
        return self
    
    def set_normed(self, data, idx = None):
        if idx is None:
            self.data = data * (1e-5 + self.std) + self.mean
        else:
            self.data[idx] = data * (1e-5 + self.std) + self.mean


class Data(object):
    def __init__(self, data, norm, shape_manager, sequences = None):
        self.base = DataBase(data, norm, shape_manager, sequences)

    def copy(self):
        if hasattr(self.base.data, "copy"):
            return Data(self.base.data.copy(), self.base.norm, self.base.shape, self.base.sequences)
        elif hasattr(self.base.data, "clone"):
            return Data(self.base.data.detach().clone(), self.base.norm, self.base.shape, self.base.sequences)
        else:
            return None

    def clone(self):
        new_clone = Data(self.base.data, self.base.norm, self.base.shape, self.base.sequences)
        return new_clone

    def __call__(self, idx):
        if len(self.base.data.shape) == 2:
            return Data(self.base.data[self.base.clips[idx]], self.base.norm, self.base.shape)
        else:
            return self

    def __getitem__(self, idx):
        if len(self.base.data.shape) == 2:
            return Data(self.base.data[idx], self.base.norm, self.base.shape)
        else:
            return self

    def __len__(self):
        return len(self.base.data)

    def __getattribute__(self, name):
        if name in ["base", "copy"]:
            return super().__getattribute__(name)
        if hasattr(self.base, name):
            return self.base.__getattribute__(name)
        return self.shape.get(self.data, name)

    def __setattr__(self, name, value):
        if name == "base":
            super().__setattr__(name, value)
        elif hasattr(self.base, name):
            return self.base.__setattr__(name, value)
        else:
            self.shape.set(self.data, name, value)
    


if __name__ == "__main__":
    # data = np.random.random((2000,10000))
    # norm = np.random.random((2,10000))
    
    data = np.load('./data/train16.npy')
    norm = np.load('./data/input_norm.npy')
    seq = np.loadtxt('./data/TrainSequences.txt')

    x = Data(data, norm, InputShapeManager())

    print (np.where(np.all((x.goal_action_labels == "Walk"), 1)))
    print (seq[0])
    # print (x.data[np.all((x.goal_action_labels == "Walk"), 1)])
    # print (x.shape.goal_action_labels[x.goal_action.argmax(-1)[20]])
