# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021

import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch
import cv2
import json
import os
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2



parser = argparse.ArgumentParser(description='cam')
parser.add_argument('--model', type=str, default='ResNet18')
parser.add_argument('--model_num', type=str, default= '7')

arg = parser.parse_args()


# input image
image_dir = './data/cam_test'

model_dir = os.path.join('log', arg.model, 'checkpoints', f'000{arg.model_num}.pth')
print(model_dir)
check_point = torch.load(model_dir)

if arg.model == "ResNet18":        
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(check_point['model'])
    # model = model.to('cuda')
    final_conv = 'layer4'

elif arg.model == "ResNet50":        
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 2)
    model.load_state_dict(check_point['model'])
    final_conv = 'layer4'
    # model = model.to('cuda')

elif arg.model == "ResNet101":        
    model = models.resnet101(pretrained=False)
    model.fc = nn.Linear(2048, 2)
    model.load_state_dict(check_point['model'])
    final_conv = 'layer4'
    # model = model.to('cuda')
elif arg.model =='se_resnet18':
    from senet.se_resnet import se_resnet18
    model = se_resnet18(num_classes=2)
    model.load_state_dict(check_point['model'])
    final_conv = 'layer4'
elif arg.model =='se_resnet34':
    from senet.se_resnet import se_resnet34
    model = se_resnet34(num_classes=2)
    model.load_state_dict(check_point['model'])
    final_conv = 'layer4'
elif arg.model =='se_resnet50':
    from senet.se_resnet import se_resnet50
    model = se_resnet50(num_classes=2)
    model.load_state_dict(check_point['model'])
    final_conv = 'layer4'
elif arg.model =='CBAM_18':
    from MODELS.model_resnet import *
    model = ResidualNet( 'ImageNet', 18, 2,  CBAM)
    model.load_state_dict(check_point['model'])
    final_conv = 'layer4'
else:

    raise NotImplementedError

model.eval()

print(model)

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

model._modules.get(final_conv).register_forward_hook(hook_feature)

# get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

transforms_image = A.Compose([
    A.Resize(224, 224),
    ToTensorV2(),
])

real_image_list = os.listdir('./data/cam_test/real')
fake_image_list = os.listdir('./data/cam_test/fake')

for i in range(len(real_image_list)):
    # load test image
    real_image = Image.open(os.path.join('./data/cam_test/real',real_image_list[i]))
    # fake_image = Image.open(os.path.join('./data/cam_test/fake',fake_image_list[i]))
    image = np.array(real_image)

    real_tensor = transforms_image(image=image)['image']
    real_tensor = real_tensor/255.0
    img_variable = Variable(real_tensor.unsqueeze(0))
    logit = model(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    print(h_x)
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    print(idx)

    if idx[0] == 1:
        predict = 'OK'
    else:
        predict = 'NO'

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: {}'.format([idx[0]]))
    img = cv2.imread(os.path.join('./data/cam_test/real',real_image_list[i]))
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.6 + img * 0.5
    cv2.imwrite('heat_map_result/{}/real/{}_{}.jpg'.
                    format(arg.model,real_image_list[i],predict), result)

for i in range(len(fake_image_list)):
    # load test image
    fake_image = Image.open(os.path.join('./data/cam_test/fake',fake_image_list[i]))
    # fake_image = Image.open(os.path.join('./data/cam_test/fake',fake_image_list[i]))
    image = np.array(fake_image)

    real_tensor = transforms_image(image=image)['image']
    real_tensor = real_tensor/255.0
    img_variable = Variable(real_tensor.unsqueeze(0))
    logit = model(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    print(h_x)
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    print(idx)

    if idx[0] == 0:
        predict = 'OK'
    else:
        predict = 'NO'

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: {}'.format([idx[0]]))
    img = cv2.imread(os.path.join('./data/cam_test/fake',fake_image_list[i]))
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.6 + img * 0.5
    cv2.imwrite('heat_map_result/{}/fake/{}_{}.jpg'.
                    format(arg.model,real_image_list[i],predict), result)