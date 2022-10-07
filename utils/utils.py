from re import L
import torch 
import numpy as np
from models.cls_train import * 
from models.backbone import *
from utils.dataloader import * 
import pickle
 
def load_attack_model(args, num_classes):
    if args.target_model.startswith("AlexNet"):
        model = get_AlexNet(args.target_model, num_classes, args.dataset)
    elif args.target_model.startswith("VGG"):
        model = get_VGG(args.target_model, num_classes, args.dataset)
    elif args.target_model.startswith("ResNet"):
        model = get_resnet(args.target_model, num_classes, args.dataset)
    
    if args.dataset != "Imagenet":
        model.load_state_dict(torch.load(os.path.join("checkpoint", "{}_{}.pth".format(args.target_model, args.dataset)) ))
    if args.dataset == "MNIST":
        model = MNIST_Model(10)
        model.load_state_dict(torch.load(os.path.join("checkpoint", "MNIST.pth")))
    return model 

def load_transfer_model(args, num_classes):
    if args.transfer_model.startswith("AlexNet"):
        model = get_AlexNet(args.transfer_model, num_classes, args.dataset)
    elif args.transfer_model.startswith("VGG"):
        model = get_VGG(args.transfer_model, num_classes, args.dataset)
    elif args.transfer_model.startswith("ResNet"):
        model = get_resnet(args.transfer_model, num_classes, args.dataset)
    
    if args.dataset != "Imagenet":
        model.load_state_dict(torch.load(os.path.join("checkpoint", "{}_{}.pth".format(args.transfer_model, args.dataset)) ))
    if args.dataset == "MNIST":
        model = MNIST_Model(10)
        model.load_state_dict(torch.load(os.path.join("checkpoint", "MNIST.pth")))
    return model 


def label_to_image_cifar(args, filename):
    cifardataset = CIFAR10Dataset(args, filename)
    label_dict = {}
    for i in range(10000):
        img, label, index = cifardataset[i]
        label = torch.argmax(label)
        if label.item() not in label_dict:
            label_dict[label.item()] = [index]
        else:
            label_dict[label.item()].append(index)
    pickle.dump(label_dict, open("data/cifar.pickle", "wb"))
