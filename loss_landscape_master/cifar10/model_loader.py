import os
import torch, torchvision
import cifar10.models.resnet as resnet

# map between model name and function
models = {
    'resnet20'              : resnet.resnet20,
    'resnet32'              : resnet.resnet32,
    'resnet44'              : resnet.resnet44,
    'resnet56'              : resnet.resnet56,
    'resnet110'             : resnet.resnet110,
}

def load(model_name, model_file=None, data_parallel=False):
    net = models[model_name]()
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'model_state_dict' in stored.keys():
            net.load_state_dict(stored['model_state_dict'])
        else:
            net.load_state_dict(stored)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
