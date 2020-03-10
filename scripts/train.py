import sys; sys.path.append('.')

import os
from tqdm import tqdm 
import numpy as np
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from src.data import MotionDataset
from src.data import MotionChunkDataset
from src.utils.watcher import Watcher
from src.nn.nets import NSM

def train_epoch(epoch, model, dataloader, optimizer, optimize_step = 4):
    global global_step
    step = 0
    train_accuracy = 0
    model.train(True)
    pbar = tqdm(dataloader)

    for sample in pbar:
        Fr, G, E, I, Ga, output = map(lambda x: x.cuda(), sample.values())

        step += 1
        global_step += 1

        output_predicted = model(Fr, G, E, I, Ga)
        train_loss = criterion(output_predicted, output)
        train_loss.backward()

        if (step + 1) % optimize_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        pbar.set_description(f'[{epoch}] Loss: {train_loss.item()}, Acc: {train_accuracy}')
        # watcher.update(global_step, locals())

def validate(epoch, model, dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():        
        for sample in tqdm(dataloader):
            Fr, G, E, I, Ga, output = map(lambda x: x.cuda(), sample.values())

            output_predicted = model(Fr, G, E, I, Ga)
            test_loss += criterion(output_predicted, output)

        # watcher.update(epoch, locals())
        
    return test_loss

def accuracy(input, target):
    with torch.no_grad():
        return (input.argmax(1).eq(target).sum().float() / input.shape[0]).item()

def save(model, path, accu_val, epoch):
    torch.save(model, f'{path}/model_{epoch}_{accu_val}.pt')

def gen_name(batch_size):
    return "models/t%d/" % (len(os.listdir('models'))+1)


if __name__ == "__main__":
    # Params, make this args
    n_experts = 10

    # Joints, Goal, Environment, Interaction
    input_shape = [432, 169, 2034, 2048]
    encoders_shape = [512,128,512,128]
    motionnet_shape = [512,512,638]
    gatingnet_shape = [754, 128]

    # Move to args
    epochs = 300
    batch_size = 1024
    global_step = 0
    save_dir = gen_name(batch_size)
    os.makedirs(save_dir)

    train_loader = DataLoader(
        MotionChunkDataset(
            'data/data16', 
            input_shape = 5437,
            num_samples = 741286,
            num_chunks = 4
        ),
        batch_size,
    )

    # val_loader = DataLoader(
    #     MotionDataset('data'),
    #     batch_size,
    # )
    # print(len(val_loader.dataset))

    # Load model
    start_epoch = 0
    model = NSM(
        input_shape,
        encoders_shape,
        motionnet_shape,
        gatingnet_shape,
        n_experts
    ).cuda().half()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    watcher = Watcher(None)

    watcher.add_scalar("train_loss")
    # watcher.add_scalar("test_loss")

    for epoch in range(start_epoch, epochs):
        train_epoch(epoch, model, train_loader, optimizer)
        # test_loss = validate(epoch, model, val_loader)
        save(model, save_dir, 0, epoch)

