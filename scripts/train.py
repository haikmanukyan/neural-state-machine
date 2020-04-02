import sys; sys.path.append('.')

import os
from tqdm import tqdm 
import numpy as np
from itertools import chain
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_


from src.data import MotionDataset
from src.utils import *
from src.nn.nets import NSM


def train_epoch(epoch, model, dataloader, optimizer):
    step = 0
    train_accuracy = 0
    model.train(True)
    pbar = tqdm(dataloader)
    global global_step

    for sample in pbar:
        Fr, G, E, I, Ga, output = map(lambda x: x.cuda(), sample.values())

        global_step += 1
        step += 1

        output_predicted = model(Fr, G, E, I, Ga)
        train_loss = criterion(output_predicted, output)
        train_loss.backward()

        # Check for exploding gradients and restart if it's present
        grad_norm = sum([p.grad.data.norm(2) for p in model.parameters()]) ** 0.5
        if grad_check(grad_norm):
            return True

        optimizer.step()
        optimizer.zero_grad()
        
        pbar.set_description(f'[{epoch}] Loss: {train_loss.item()}, Acc: {train_accuracy}')
        if step % 40 == 0:
            grad_mean = grad_check.grad_mean
            watcher.update(global_step, locals())
    return False

def validate(epoch, model, dataloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        pbar = tqdm(dataloader)   
        for sample in pbar:
            Fr, G, E, I, Ga, output = map(lambda x: x.cuda(), sample.values())

            output_predicted = model(Fr, G, E, I, Ga)
            test_loss += criterion(output_predicted, output)
        test_loss /= len(dataloader)
        pbar.set_description(f'Test Loss: {test_loss}')
        watcher.update(epoch, locals())
        
    return test_loss

def accuracy(input, target):
    with torch.no_grad():
        return (input.argmax(1).eq(target).sum().float() / input.shape[0]).item()

def gen_name():
    i = 0
    existing = "(".join(os.listdir("models"))
    while "model%d(" % (i + 1) in existing:
        i += 1
    return "models/model%d" % (i + 1)

def load_model(model_name):
    for path in os.listdir("models"):
        if path.startswith(model_name + "("):
            models = os.listdir("models/" + path)
            last_model = None
            for x in models: 
                if x.startswith("epoch"): last_model = x
            epoch = int(last_model.split("_")[1].split("-")[0])
            return torch.load("models/" + path + "/" + last_model), np.load("models/" + path + "/loss.npy"), epoch
    raise Exception("No model with name " + model_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Use to train a network. 
    The network will be stored in the models directory with an automatic name. 
    Run from the project root. Remember to start visdom to see the logs! 
    Install ipdb if you want to debug your code after keyboard interrupts''')
    # Training related args
    parser.add_argument("--load-model", type = str, default = None, help = "Name of the model you want to load from the models directory. Must be the folder name, ignoring the parenthesis")
    parser.add_argument("--comment", type = str, default = "Batch Norm", help = "A description to store with the trained network")
    
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Initial learning rate")
    parser.add_argument("--lr-alpha", type = float, default = 0.75, help = "Value to multiply the learning rate with after a restart")
    parser.add_argument("--batch-size", type = int, default = 1024, help = "Batch size")
    parser.add_argument("--n-epochs", type = int, default = 100, help = "Number of epochs")
    parser.add_argument("--restart-threshold", type = int, default = 10, help = "The multiplicative threshold for the gradient norm to decide restarts")


    # Network related args
    parser.add_argument("--input-size", type = int, default=5437, help = "Number of input dimensions")
    parser.add_argument("--n-experts", type = int, default=10, help = "Number of experts")
    parser.add_argument("--input-shape", type=str, default="432,169,2034,2048", help = "Dimensions of each part of the input (frame, goal, environment, interaction)")
    parser.add_argument("--encoders-shape", type=str, default="512,128,512,256", help = "Dimensions of the hidden layer in each of the encoders")
    parser.add_argument("--motionnet-shape", type=str, default="512,512,638", help = "The architecture of the motion prediction network")
    parser.add_argument("--gatingnet-shape", type=str, default="754,128", help = "The architecture of the gating network")

    args = parser.parse_args()

    # Split data into
    # Joints, Goal, Environment, Interaction
    input_shape = str2list(args.input_shape)
    encoders_shape = str2list(args.encoders_shape)
    motionnet_shape = str2list(args.motionnet_shape)
    gatingnet_shape = str2list(args.gatingnet_shape)

    # Initialize
    global_step = 0
    save_dir = gen_name()
    os.makedirs(save_dir)
    
    saver = ModelSaver(save_dir)
    grad_check = GradientCheck(args.restart_threshold)
    watcher = Watcher(None)

    watcher.add_scalar("train_loss")
    watcher.add_scalar("test_loss")
    watcher.add_scalar("grad_norm")
    watcher.add_scalar("grad_mean")

    # Load model
    if args.load_model == None:
        model = NSM(
            input_shape,
            encoders_shape,
            motionnet_shape,
            gatingnet_shape,
            args.n_experts
        ).cuda()
        start_epoch = 0
    else:
        model, saved_loss, start_epoch = load_model(args.load_model)

        # Load tracked loss into memory, for better visualization
        for epoch,test_loss in enumerate(saved_loss):
            saver.history.append(test_loss)
            watcher.update(epoch, locals())
    print (model)

    # Load Data
    train_loader = DataLoader(
        MotionDataset(
            'data/train32.npy', 
            input_shape = args.input_size,
            input_norm = np.load('data/input_norm.npy'),
            output_norm = np.load('data/output_norm.npy')
        ),
        args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        MotionDataset(
            'data/test32.npy',
            input_shape = args.input_size,
            input_norm = np.load('data/input_norm.npy'),
            output_norm = np.load('data/output_norm.npy')
        ),
        1024,
    )

    # Initialize Training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    
    try:
        for epoch in range(start_epoch, start_epoch + args.n_epochs):
                
            if train_epoch(epoch, model, train_loader, optimizer):
                print ("Gradient exploded at epoch %d, restarting" % epoch)
                optimizer.zero_grad()
                grad_check.reset()
                args.lr *= args.lr_alpha
                optimizer = optim.AdamW(model.parameters(), lr = args.lr)
            else:
                test_loss = validate(epoch, model, val_loader)
                saver.save(model, test_loss.item())
    except KeyboardInterrupt:
        # Spawn a debug console
        os.rename(save_dir, save_dir + '(%.4f)' % test_loss)
        print ("Saving model at", save_dir)
        try: import ipdb; ipdb.set_trace()
        except: print ("Install ipdb if you want to debug your model in the end")