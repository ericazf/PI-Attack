import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import models, transforms 
import torch.nn as nn 
from PIL import Image 
from models.backbone import * 

class Trainer(object):
    def __init__(self, args, dataloader, testloader):
        super(Trainer, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.dataset = args.dataset
        self.save_name = "{}_{}.pth".format(args.target_model, self.dataset)
        data_classes = {"CIFAR-10":10, "MNIST":10, "Imagenet":1000}

        if args.target_model.startswith("AlexNet"):
            self.model = get_AlexNet(args.target_model, data_classes[args.dataset], args.dataset).cuda()
        elif args.target_model.startswith("VGG"):
            self.model = get_VGG(args.target_model, data_classes[args.dataset], args.dataset).cuda()
        elif args.target_model.startswith("ResNet"):
            self.model = get_resnet(args.target_model, data_classes[args.dataset]).cuda()
 
        if args.dataset == "MNIST":
            self.model = MNIST_Model(0).cuda()
            self.save_name = "MNIST.pth"

    def train(self, epochs):
        CEloss = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
        for epoch in range(epochs):
            print("epoch: ", epoch)
            for i, (images, labels, index) in enumerate(self.dataloader):
                images = images.cuda()
                labels = torch.argmax(labels, dim = 1)
                labels = labels.cuda()
                _, preds = self.model(images)

                loss = CEloss(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 1000 == 0:
                    print("step: {}, loss: {:.5f}".format(i, loss))
            self.save_model()
            self.test()
        
    def test(self):
        CEloss = nn.CrossEntropyLoss()
        self.load_model()
        self.model.eval()
        num = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for i, (images, labels, index) in enumerate(self.testloader):
                images = images.cuda()
                labels = torch.argmax(labels, dim = 1)
                _, preds = self.model(images)
                loss = CEloss(preds, labels.cuda())
                preds = torch.argmax(preds, dim = 1)
                for pred, label in zip(preds, labels):
                    total = total + 1
                    if pred == label:
                        num = num + 1
                total_loss = total_loss + loss
        acc = num / total
        print("{}:acc {:.5f}, loss:{:.5f}".format(self.save_name, acc, total_loss/total))
    
    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join("checkpoint", self.save_name))
    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join("checkpoint", self.save_name)))






        
