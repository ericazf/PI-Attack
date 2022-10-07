import torch 
import torch.nn as nn 
import numpy as np 
import os 
from PIL import Image 
from torchvision import transforms
from utils.dataloader import * 
import pickle

class PI_NES(object):
    def __init__(self, model, args):
        self.target_model = model
        self.args = args   
        self.plateau_length = 5
        self.plateau_drop = 2
        self.max_lr = 1e-2
        self.min_lr = 5e-5
        self.batch_size = 50
        self.max_queries = 10000
        self.max_iters = int(self.max_queries/self.batch_size)
        
        self.starting_eps = 1.0
        self.starting_delta_eps = 0.5
        self.goal_epsilon = 0.05
        
        self.adv_thresh = 0.2
        self.sigma = 1e-3
        self.momentum = 0.9
        self.conservative = 2

        self.zero_iters = 5
        self.label_only_sigma = 2e-3
        
        #1:standard 2:partial_info 3:label_only
        self.standard = False
        self.partial_info = True
        self.label_only = False
        if self.standard:
            self.type = 1
        elif self.partial_info:
            self.type = 2
        elif self.label_only:
            self.type = 3
        self.top_k = 3
        
    
    def __call__(self, query, target_label, targeted):
        adv_query = self.train(query, target_label, True)
        return adv_query 

    def get_grad(self, query, target_label, target = True):
        adv_query = query.cuda()
        target_label = target_label.cuda()

        #gradient estimation
        if self.label_only:
            loss_fn = self.label_only_loss  # label only 
        elif self.partial_info:
            loss_fn = self.partial_info_loss # partial logits 
        elif self.standard:
            loss_fn = self.standard_loss  #all the logits
        
        noise_pos = torch.randn(self.batch_size//2, adv_query.size(1), adv_query.size(2), adv_query.size(3)).cuda()
        noise = torch.cat((noise_pos, -noise_pos), dim = 0)
        eval_points = adv_query + self.sigma * noise 
        losses, noise = loss_fn(eval_points, target_label, noise)

        losses = losses.view(losses.size(0), 1, 1, 1)
        grad = torch.mean(losses * noise, dim = 0)/self.sigma
        return torch.mean(losses), grad 

    def STOP(self, adv_query, target_label):
        adv_query = adv_query.cuda()
        target_label = target_label.cuda()
        _, logits = self.target_model(adv_query)
        preds = torch.argmax(logits, dim = 1)
        target = torch.argmax(target_label, dim = 1)
        if preds.item() == target.item():
            return True 
        else:
            return False

    def train(self, query, target_label, target = True):       
        #===========parameter setup================== 
        query = query.cpu()
        target_label = target_label.cpu()
        max_lr = self.max_lr 
        epsilon = self.starting_eps
        delta_epsilon = self.starting_delta_eps
        if self.type == 1:
            epsilon = self.goal_epsilon
        num_queries = 0
        last_ls = []
        #===========initial image====================
        ori_query = query
        if self.standard:
            adv_query = ori_query 
        else:
            adv_query, _ = self.image_of_class(target_label)
        lower = torch.clamp(ori_query - epsilon, min = 0, max = 1)
        upper = torch.clamp(ori_query + epsilon, min = 0, max = 1)
        adv_query = np.clip(adv_query, lower, upper)

        for i in range(300):
            if epsilon <= self.goal_epsilon and self.STOP(adv_query, target_label): 
                break 
            loss, grad = self.get_grad(adv_query, target_label)
            
            # learning rate adjustment is the key issue
            last_ls.append(loss)
            last_ls = last_ls[-self.plateau_length:]
            if last_ls[-1] > last_ls[0] and len(last_ls) == self.plateau_length: 
                if max_lr > self.min_lr:
                    max_lr = max(max_lr/self.plateau_drop, self.min_lr) 
                last_ls = [] 

            current_lr = max_lr 
            prop_de = 0.0
            if loss < self.adv_thresh and epsilon > self.goal_epsilon:
                prop_de = delta_epsilon
            
            while current_lr >= self.min_lr:
                if self.type != 1:
                    tmp_epsilon = max(epsilon - prop_de, self.goal_epsilon)
                    lower = torch.clamp(ori_query - tmp_epsilon, min = 0, max = 1)
                    upper = torch.clamp(ori_query + tmp_epsilon, min = 0, max = 1)
                adv_tmp = adv_query - current_lr * grad.cpu().sign()
                adv_tmp = np.clip(adv_tmp, lower, upper)

                if self.robust_in_top_k(adv_tmp, target_label):
                    if prop_de > 0:
                        delta_epsilon = max(prop_de, 0.1)
                        max_lr = self.max_lr 
                        last_ls = []
                    adv_query = adv_tmp 
                    epsilon = max(epsilon - prop_de, self.goal_epsilon) 
                    break 
                elif current_lr >= self.min_lr*2:
                    current_lr = current_lr/2
                else:
                    prop_de = prop_de/2
                    # print("=======", prop_de)
                    if prop_de == 0:
                        raise ValueError("Did not converge")
                    if prop_de < 2e-3:
                        prop_de = 0
                    current_lr = self.max_lr
            print("-------------iter:{} eps:{:.5f}, loss:{:.5f}, current_lr:{}, max_lr:{}".format(i, epsilon, loss, current_lr, max_lr))  
            
        return adv_query.cuda()

    def robust_in_top_k(self, adv_query, target_class):
        if self.type == 1:
            return True
        with torch.no_grad():
            _, logits = self.target_model(adv_query.cuda())
        target = torch.argmax(target_class)
        value, index = torch.sort(logits.cpu(), descending = True, dim = 1)
        index = index[:, 0:self.top_k]
        if target in index:
            return True
        else:
            return False 

    def standard_loss(self, input, label, noise):
        label = torch.argmax(label)
        label = label.expand(input.size(0))
        celoss = nn.CrossEntropyLoss(reduction = "none")
        with torch.no_grad():
            _, logits = self.target_model(input)
        loss = celoss(logits, label)
        return loss, noise

    def partial_info_loss(self, input, target_label, noise):
        target = torch.argmax(target_label, dim = 1)
        target_label = target.expand(input.size(0))
        celoss = nn.CrossEntropyLoss(reduction = "none")
        with torch.no_grad():                                                                                                                                                                   
            _, logits = self.target_model(input)
        losses = celoss(logits, target_label)
        value, index = torch.sort(logits, dim = 1, descending = True)
        index = index[:, 0:self.top_k]
        loc = torch.where(index == target)
        good_images = loc[0]

        losses = losses[good_images]
        noise = noise[good_images]
        return losses, noise  

    def label_only_loss(self, input, target_label, noise):
        target_idx = torch.argmax(target_label, dim = 1)[0]
        tiled_points = input.unsqueeze(0) #1 50 c w h
        one = torch.ones(self.zero_iters, 1,1,1,1).cuda() #
        tiled_points = tiled_points * one 
        
        noised_eval_im = tiled_points + (torch.rand(tiled_points.size())*2-1).cuda()*self.label_only_sigma
        noised_eval_im = noised_eval_im.view(-1, tiled_points.size(-3), tiled_points.size(-2), tiled_points.size(-1))
        with torch.no_grad():
            _, logits = self.target_model(noised_eval_im)
        value, index = torch.sort(logits, dim = 1, descending = True)
        index = index.view(self.zero_iters, self.batch_size, -1)
        index = index[:,:,0:self.top_k]

        rank_range = torch.range(start = self.top_k, end = 1, step = -1)
        rank_range = rank_range.view(1, 1, self.top_k)
        tiled_rank_range = rank_range * torch.ones(self.zero_iters, self.batch_size, 1)
        batch_rank = torch.where(index==target_idx, tiled_rank_range.cuda(), torch.zeros(tiled_rank_range.size()).cuda())
        batch_rank = torch.max(batch_rank, dim = 2)[0].to(torch.float32)
        proxy_score = self.top_k - torch.mean(batch_rank, dim = 0)
        return proxy_score, noise       

    def image_of_class(self, target_label):
        target = torch.argmax(target_label)
        if self.args.dataset == "CIFAR-10":
            label_index = pickle.load(open("data/cifar.pickle", "rb"))
            list = label_index[target.item()]
            s = random.sample(range(0,len(list)),1)[0]
            index = list[s]
            cifardataset = CIFAR10Dataset(self.args, ["test_batch"])
            img, label, idx = cifardataset[index]
        
        return img, label 
        