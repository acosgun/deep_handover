#!/usr/bin/env python2
import visual_servoing_dataset
import torch
import torch.nn as nn
import model
import torch.optim as optim
import dataset_slice
import random
import matplotlib.pyplot as plt
import numpy as np

class ControlLoss():

    def __init__(self):
        self.cos_sim = nn.CosineSimilarity()
        self.p_dist = nn.PairwiseDistance()

    def __call__(self,input1,input2):

        input1_v = input1[:,0:3]
        input1_r = input1[:,3:6]

        input2_v = input2[:,0:3]
        input2_r = input2[:,3:6]

        norm1_v = torch.norm(input1_v,p=2,dim=1)
        norm1_r = torch.norm(input1_r,p=2,dim=1)
        norm2_v = torch.norm(input2_v,p=2,dim=1)
        norm2_r = torch.norm(input2_r,p=2,dim=1)


        cos_sim_v = 1.0-self.cos_sim(input1_v,input2_v)
        cos_sim_v = cos_sim_v * (norm1_v > 0.01).float() * (norm2_v > 0.01).float()

        dist_v = self.p_dist(input1_v,input2_v)
        dist_v = torch.squeeze(dist_v)


        cos_sim_r = 1.0-self.cos_sim(input1_r,input2_r)
        cos_sim_r = cos_sim_r * (norm1_r > 0.01).float() * (norm2_r > 0.01).float()
        dist_r = self.p_dist(input1_r,input2_r)
        dist_r = torch.squeeze(dist_r)

        loss_v = 0.9*cos_sim_v + 0.1 * dist_v
        loss_r = 0.9*cos_sim_r + 0.1 * dist_r

        loss_joined = loss_v + loss_r
        loss = torch.mean(loss_joined)

        return loss

def main():
    #load the dataset
    dataset = visual_servoing_dataset.Dataset(["/home/acrv/data/visual_servoing/handover"])
    #shuffle the dataset
    random.shuffle(dataset.img_annotation_path_pairs)

    n = len(dataset)
    r = 0.8
    i1 = int(n*r)

    train_data = dataset_slice.DataSlice(dataset,0,i1)
    test_data = dataset_slice.DataSlice(dataset,i1,n)

    print(len(dataset),len(train_data),len(test_data))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers = 4)
    net = model.ResNet()
    #net = model_2.Net()
    # net = nednet.Net()
    state_dict = torch.load('resnet18-5c106cde.pth')
    net.load_partial_state_dict(state_dict)

    train(net,train_loader,test_loader)

def train(net,train_loader,test_loader):
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []

    net.cuda()
    net.train()

    #criterion = ControlLoss()
    criterion = nn.BCELoss()
    #criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # get the inputs
            img = torch.autograd.Variable(data["image"]).cuda()
            forces = torch.autograd.Variable(data["force"]).cuda()
            labels = torch.autograd.Variable(data["annotation"]).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(img,forces)
            loss = criterion(outputs, labels)

            # print("output",outputs)
            # print("loss",loss)

            # criterion2(outputs, labels)


            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += float(loss.data)

            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] batch loss: %0.5f' %(epoch + 1, i + 1, loss.data))

        train_loss = running_loss / i
        test_loss, test_accuracy = test(net,test_loader,criterion)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_accuracy)

        plt.cla()
        plt.plot(train_loss_list)
        plt.plot(test_loss_list)
        plt.plot(test_acc_list)
        plt.draw()
        plt.pause(0.1)

        print("Train/Test %0.5f / %0.5f" % (train_loss,test_loss))

        torch.save(net,"models/handover_2.pt")
    print('Finished Training')
    plt.show()

def test(net,test_loader,criterion):
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for i, data in enumerate(test_loader, 0):

        # get the inputs
        img = torch.autograd.Variable(data["image"]).cuda()
        forces = torch.autograd.Variable(data["force"]).cuda()
        labels = torch.autograd.Variable(data["annotation"]).cuda()

        # forward + backward + optimize
        outputs = net(img,forces)
        loss = criterion(outputs, labels)

        output_thresh = outputs.cpu().data.numpy() > 0.5
        #from torch.autograd import Variable
        #output_thresh   = (Variable(outputs).data).cpu().numpy()
        correct = output_thresh == labels.cpu().data.numpy()
        # print(correct)
        correct_sum = np.sum(correct)

        running_correct += correct_sum
        running_total = running_total + len(output_thresh)

        # print statistics
        running_loss += float(loss.data)

    test_loss = running_loss / i
    test_accuracy = running_correct / float(running_total)
    print(running_correct,running_total)

    return test_loss, test_accuracy


if __name__ == "__main__":
    main()
