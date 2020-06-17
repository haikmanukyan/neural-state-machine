import sys
import argparse
import torch
sys.path.append('.')

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default = 'models/best.pt', help = "The name of the model you want to test, just give the folder name ignoring the parenthesis")
args = parser.parse_args()

model = torch.load(args.model_name).cuda()
model.eval()

432,169,2034,2048
dummy1 = torch.zeros((1,432)).cuda()
dummy2 = torch.zeros((1,169)).cuda()
dummy3 = torch.zeros((1,2034)).cuda()
dummy4 = torch.zeros((1,2048)).cuda()
dummy5 = torch.zeros((1,754)).cuda()

torch.onnx.export(model, (dummy1,dummy2,dummy3,dummy4, dummy5), 'models/model.onnx', verbose=True)