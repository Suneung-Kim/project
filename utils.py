import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm
from MODELS.model_resnet import *

def get_network(
    options
):
    model = None

    if options.network == "ResNet":        
        model = models.resnet18(pretrained=options.model.pretrained)
        model.fc = nn.Linear(512, options.data.num_classes)
    
    elif options.network == "ResNet50":        
        model = models.resnet50(pretrained=options.model.pretrained)
        model.fc = nn.Linear(2048, options.data.num_classes)
    elif options.network == "ResNet101":        
        model = models.resnet101(pretrained=options.model.pretrained)
        print(model)
        model.fc = nn.Linear(2048, options.data.num_classes)
    elif options.network == "se_resnet18":
        from senet.se_resnet import se_resnet18
        model = se_resnet18(num_classes=2)
    elif options.network == "se_resnet34":
        from senet.se_resnet import se_resnet34
        model = se_resnet34(num_classes=2)
    elif options.network == "se_resnet50":
        from senet.se_resnet import se_resnet50
        model = se_resnet50(num_classes=2)
    elif options.network == "cbam_50":
        model = ResidualNet( 'ImageNet', 50, 2,  BAM)
    else:
        raise NotImplementedError

    return model.to(options.device)

def get_optimizer(
    params,
    options
):
    if options.optimizer.type == "Adam":
        optimizer = optim.Adam(params, lr=options.optimizer.lr, weight_decay=options.optimizer.weight_decay)
    else:
        raise NotImplementedError

    return optimizer
