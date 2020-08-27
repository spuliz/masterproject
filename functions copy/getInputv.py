import torch
from torch.autograd import Variable

device = torch.device("cpu")

def get_inputv(inp):
    input_stack = torch.FloatTensor().to(device)
    input_stack.resize_as_(inp.float()).copy_(inp)
    inputv = Variable(input_stack)
    return inputv