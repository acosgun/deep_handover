#!/usr/bin/env python

import torch
import torch.nn as nn
import math
import torchvision
import torch.utils.model_zoo as model_zoo
import time
import numpy as np
import cv2


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block = BasicBlock, layers = [2,2,2,2]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv2 = nn.Conv2d(512,16,kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 7 * 7 * block.expansion + 6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        #Overwrite the default random weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img, forces):
        x = self.bn0(img)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)

        x = x.view(x.size(0), -1)

        x = torch.cat((x,forces),dim=1)

        # x = self.fc1(x)
        # x = self.sigmoid(x)
        # x = self.fc2(x)
        # x = self.sigmoid(x)
        # x = self.fc3(x)
        # x = self.sigmoid(x)
        # x = self.fc4(x)
        # x = self.tanh(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

    def load_partial_state_dict(self,pretrained_dict):
        """This will only load weights for operations that exist in the model
        and in the state_dict"""

        state_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in state_dict}
        # 2. overwrite entries in the existing state dict
        state_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(state_dict)




if __name__ == "__main__":
    print("Hello There!")

    net = ResNet()
    print(net)
    state_dict = torch.load('resnet18-5c106cde.pth')
    # print(state_dict.keys())
    # print(type(state_dict))
    # raw_input()
    net.load_partial_state_dict(state_dict)
    #net.load_state_dict(state_dict)


    net.eval()
    net.cuda()

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    n = 1
    t1 = time.time()
    for i in range(n):
        img = torch.autograd.Variable(torch.rand(1,3,224,224)).float().cuda()
        out = net(img)
        out = out.data.cpu().numpy()
        label = np.argmax(out)
        print("Frame %i: %i" % (i,label))
    t2 = time.time()

    fps = n / (t2-t1)
    print("fps",fps)

    weight_sum = 0
    for name,parameter in net.named_parameters():
        weights = np.prod(parameter.shape)
        weight_sum += weights
        print(name,weights)
    print("total weights",weight_sum)

    # try:
    #     cap = cv2.VideoCapture(0)
    #     n = 10
    #     fps = 0
    #     while(True):
    #         t1 = time.time()
    #
    #         for i in range(n):
    #             # Capture frame-by-frame
    #             ret, img = cap.read()
    #             frame = cv2.resize(img,(224,224))
    #             frame = np.transpose(frame,(2,0,1))
    #             frame = frame.astype(np.float) / 255.0
    #
    #
    #             frame = torch.from_numpy(frame).float().cuda()
    #             frame = normalize(frame)
    #             frame = torch.autograd.Variable(frame).float().cuda()
    #             frame = frame.unsqueeze(0)
    #             out = net(frame)
    #             out = out.data.cpu().numpy()
    #             label = np.argmax(out)
    #             print("FPS %2.1f: %s" % (fps,str(out)))
    #
    #             # Display the resulting frame
    #             cv2.imshow('frame',img)
    #
    #             if cv2.waitKey(10) & 0xFF == ord('q'):
    #                 break
    #         t2 = time.time()
    #         fps = n / (t2-t1)
    #
    #
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     # When everything done, release the capture
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     print("All Closed")
