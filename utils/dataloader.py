import torch 
import torch.utils.data as data
import torchvision
from torchvision import transforms
import pickle 
import torch.nn.functional as F 
import os 
import pickle
import numpy as np 
from PIL import Image
import random 

class CIFAR10Dataset(data.Dataset):
    def __init__(self, args, filename):
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        self.file_list = []
        for name in filename:
            data_path = os.path.join(args.root, name)
            with open(data_path, "rb") as f:
                filedict = pickle.load(f, encoding = "bytes")
                self.file_list.append(filedict)
        
        self.data = np.array([])
        self.label = np.array([])
        for batch in self.file_list:
            if len(self.data) == 0:
                self.data = batch[b'data']
                self.label = batch[b'labels']
            else:
                self.data = np.append(self.data, batch[b'data'], axis = 0)
                self.label = np.append(self.label, batch[b'labels'], axis = 0)
        self.data = np.reshape(self.data, (-1, 3, 32, 32))
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        
        self.label_onehot = []
        for i in range(np.shape(self.label)[0]):
            onehot = F.one_hot(torch.LongTensor([self.label[i]]), num_classes = 10).tolist()
            self.label_onehot.append(onehot[0])
        self.label_onehot = torch.FloatTensor(self.label_onehot)
    
    def __getitem__(self, index):
        image = self.data[index]
        image = Image.fromarray(image.astype('uint8')).convert("RGB")
        label = self.label_onehot[index]
        image = self.transform(image)
        return image, label, index
    
    def __len__(self):
        return len(self.label_onehot)

    def load_label(self):
        return self.label_onehot

class CIFAR10SelectDataset(data.Dataset):
    def __init__(self, args, filename):
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        self.file_list = []
        for name in filename:
            data_path = os.path.join(args.root, name)
            with open(data_path, "rb") as f:
                filedict = pickle.load(f, encoding = "bytes")
                self.file_list.append(filedict)
        
        self.data = np.array([])
        self.label = np.array([])
        for batch in self.file_list:
            if len(self.data) == 0:
                self.data = batch[b'data']
                self.label = batch[b'labels']
            else:
                self.data = np.append(self.data, batch[b'data'], axis = 0)
                self.label = np.append(self.label, batch[b'labels'], axis = 0)
        self.data = np.reshape(self.data, (-1, 3, 32, 32))
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        
        self.label_onehot = []
        for i in range(np.shape(self.label)[0]):
            onehot = F.one_hot(torch.LongTensor([self.label[i]]), num_classes = 10).tolist()
            self.label_onehot.append(onehot[0])
        self.label_onehot = torch.FloatTensor(self.label_onehot)
        select_index = np.loadtxt("checkpoint/select.txt", dtype = int)
        self.data = self.data[select_index]
        self.label_onehot = self.label_onehot[select_index]

    def __getitem__(self, index):
        image = self.data[index]
        image = Image.fromarray(image.astype('uint8')).convert("RGB")
        label = self.label_onehot[index]
        image = self.transform(image)
        return image, label, index
    
    def __len__(self):
        return len(self.label_onehot)

    def load_label(self):
        return self.label_onehot

class MNISTDataset(data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, train = True):
        super(MNISTDataset, self).__init__()
        self.transform = transform 
        self.target_transform = target_transform 
        self.processed_folder = "MNIST/processed"
        if train:
            data_file = "training.pt"
        else:
            data_file = "test.pt"
        self.data, self.targets = torch.load(os.path.join(root, self.processed_folder, data_file))
        
        if not train:
            random.seed(10)
            index = random.sample(range(0,10000), 100)
            self.data = self.data[index]
            self.targets = self.targets[index]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode = "L")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target_onehot = F.one_hot(torch.LongTensor([target]), num_classes = 10)[0]
        return img, target_onehot, index
    
    def __len__(self):
        return len(self.targets)
 
class ImagenetDataset(data.Dataset):
    def __init__(self, image_list, label_list):
        super(ImagenetDataset, self).__init__()
        with open(image_list) as f:
            image_list = f.readlines()
        with open(label_list) as f:
            label_list = f.readlines()

        self.images = [image.strip() for image in image_list]
        self.labels = [int(label) for label in label_list]
        
        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = img.convert("RGB")
        img = self.transform(img)
        label = self.labels[index]
        label = F.one_hot(torch.tensor([label]), num_classes = 1000)[0]
        label = label.to(torch.float32)
        return img, label, index
    
    def __len__(self):
        return len(self.images)

