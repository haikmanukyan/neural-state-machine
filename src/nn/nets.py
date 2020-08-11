import torch
from torch import nn
import torch.nn.functional as F


def GatingNet(in_dims, h_dims, n_experts):
    return nn.Sequential(
        nn.Linear(in_dims, h_dims),
        # nn.Dropout(0.7),
        nn.BatchNorm1d(h_dims),
        nn.ELU(),
        nn.Linear(h_dims, h_dims),
        nn.BatchNorm1d(h_dims),
        # nn.Dropout(0.7),
        nn.ELU(),
        nn.Linear(h_dims, n_experts),
        nn.Softmax(-1)
    )

def Encoder(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        # nn.Dropout(0.7),
        nn.BatchNorm1d(out_dim),
        nn.ELU(),
        nn.Linear(out_dim, out_dim),
        # nn.Dropout(0.7),
        nn.BatchNorm1d(out_dim),
        nn.ELU(),
        nn.Linear(out_dim, out_dim)
    )

class MotionNet(nn.Module):
    def __init__(
            self,
            input_shape,
            encoders_shape,
            network_shape,
            n_experts,
        ):

        super(MotionNet, self).__init__()

        self.n_experts = n_experts

        self.frame_encoder = Encoder(input_shape[0], encoders_shape[0])
        self.goal_encoder = Encoder(input_shape[1], encoders_shape[1])
        self.environment_encoder = Encoder(input_shape[2], encoders_shape[2])
        self.interaction_encoder = Encoder(input_shape[3], encoders_shape[3])

        self.l1 = nn.Linear(sum(encoders_shape), network_shape[0] * self.n_experts)
        self.bn1 = nn.BatchNorm1d(network_shape[0] * self.n_experts)
        
        self.l2 = nn.Linear(network_shape[0], network_shape[1] * self.n_experts)
        self.bn2 = nn.BatchNorm1d(network_shape[1] * self.n_experts)
        
        self.l3 = nn.Linear(network_shape[1], network_shape[2] * self.n_experts)


    def forward(self, Fr, G, E, I, alpha):
        x = torch.cat([
            self.frame_encoder(Fr),
            self.goal_encoder(G),
            self.environment_encoder(E),
            self.interaction_encoder(I)
        ], 1)
        batch_size = Fr.shape[0]

        x = self.l1(x)
        x = self.bn1(x)
        x = (x.view(batch_size, -1, self.n_experts) * alpha[:,None])
        x = x.sum(-1)
        x = F.elu(x)

        x = self.l2(x)
        x = self.bn2(x)
        x = (x.view(batch_size, -1, self.n_experts) * alpha[:,None])
        x = x.sum(-1)
        x = F.elu(x)

        x = self.l3(x)
        x = (x.view(batch_size, -1, self.n_experts) * alpha[:,None])
        x = x.sum(-1)

        return x

class NSM(nn.Module):
    def __init__(
            self,
            input_shape,
            encoders_shape,
            motionnet_shape,
            gatingnet_shape,
            n_experts,
        ):
        super(NSM, self).__init__()

        self.input_shape = input_shape
        self.motionnet = MotionNet(input_shape, encoders_shape, motionnet_shape, n_experts)
        self.gatingnet = GatingNet(gatingnet_shape[0], gatingnet_shape[1], n_experts)

    def forward(self, x):
        Fr,G,E,I,Ga = torch.split(x, self.input_shape, 1)
        return self.motionnet(Fr, G, E, I, self.gatingnet(Ga))