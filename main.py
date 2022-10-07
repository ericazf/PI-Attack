import torch 
import torch.optim
import torch.utils.data 
import argparse 
import os
import glob
import torch.nn.functional as F 

from torch.utils.data.dataloader import DataLoader 
import torchvision.transforms as transforms 

from utils.dataloader import *
from utils.utils import *
from models.cls_train import * 
from models.backbone import * 
from blackbox.bb_train import * 

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

parser = argparse.ArgumentParser(description = "Adversarial Examples for Hamming Space Search HAG")
parser.add_argument("--dataset", dest = "dataset", default = "CIFAR-10", choices = ["MNIST", "CIFAR-10", "Imagenet"], help = "datasets name")
parser.add_argument("--root", dest = "root", default = "../../datasets/cifar-10", help = "root path of your data")
parser.add_argument("--data_path", dest = "data_path", default = "./data", help = "the path of dataset information")
parser.add_argument("--model_path", dest = "model_path", default = "checkpoint", help = "the path of classification models")
parser.add_argument("--target_model", dest = "target_model", default = "VGG11", choices = ["AlexNet", "VGG11", "ResNet18", "ResNet50"], help = "the name of hashing method")
parser.add_argument("--transfer_model", dest = "transfer_model", default = "VGG11", choices = ["AlexNet", "VGG11", "ResNet18"])

parser.add_argument("--batch_size", dest = "batch_size", type = int, default = 24)
parser.add_argument("--query_bs", dest = "query_bs", type = int, default = 1)
parser.add_argument("--eps", dest = "eps", default = 0.032, type = float)
parser.add_argument("--targeted", dest = "targeted", default = True, choices = [True, False])

parser.add_argument("--cls", dest = "cls", choices = [True, False], default = False)
parser.add_argument("--black_box", dest = "black_box", choices = [True, False], default = True)
args = parser.parse_args()

if args.dataset == "CIFAR-10":
    train_filename = ["data_batch_1", "data_batch_2"]
    query_filename = ["test_batch"]
    train_dataset = CIFAR10Dataset(args, train_filename)
    trainloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers=4)
    test_dataset = CIFAR10Dataset(args, query_filename)
    testloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers=4)

    query_dataset = CIFAR10SelectDataset(args, query_filename)
    queryloader = DataLoader(query_dataset, batch_size = args.query_bs, shuffle = False, num_workers=4)
    unique_labels = query_dataset.load_label().unique(dim = 0)
    num_query = len(query_dataset)

elif args.dataset == "MNIST":
    train_dataset = MNISTDataset(root = "data", transform = transforms.ToTensor(), train = True)
    trainloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers=4)
    test_dataset = MNISTDataset(root = "data", transform = transforms.ToTensor(), train = False)
    testloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers=4)

    queryloader = DataLoader(test_dataset, batch_size = args.query_bs, shuffle = False, num_workers = 4)

elif args.dataset == "Imagenet":
    query_dataset = ImagenetDataset("data/Imagenet/query_img.txt", "data/Imagenet/query_label.txt") 
    queryloader = DataLoader(query_dataset, batch_size = args.query_bs, shuffle = False, num_workers = 4)

data_classes = {"CIFAR-10":10, "MNIST":10, "Imagenet":1000}
num_classes = data_classes[args.dataset]
 
if args.cls == True:
    if args.dataset == "Imagenet":
        print("target model: {}_{}".format(args.target_model, args.dataset))
        trainer = Trainer(args, queryloader, queryloader) 
        trainer.load_model()
        trainer.test()
    else:
        print("target model: {}_{}".format(args.target_model, args.dataset))
        trainer = Trainer(args, trainloader, testloader) 
        # trainer.train(args.epochs)
        trainer.load_model()
        trainer.test()

if args.black_box:
    attack_model = load_attack_model(args, num_classes)
    transfer_model = load_transfer_model(args, num_classes)
    attacker = Blackbox(parser, attack_model, transfer_model, queryloader)
    attacker.train()




