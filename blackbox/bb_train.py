import torch 
import torch.nn as nn 
import argparse 
from blackbox.PI_NES import * 

def parameter(parser):
    parser.add_argument("--type", dest = "type", default = "linf", choices = ["l0", "l2", "linf"])
    parser.add_argument("--blackbox_attack", dest = "blackbox_attack", default = "PI_NES", choices = ["PI_NES"])
    parser.add_argument("--sigma", dest = "sigma", default = 1e-3, type = float)
    parser.add_argument("--sample_num", dest = "sample_num", default = 50, type = int)
    args = parser.parse_args()

    return args 

class Blackbox(object):
    def __init__(self, parser, attack_model, transfer_model, dataloader):
        super(Blackbox, self).__init__()
        self.args = parameter(parser)
        self.dataloader = dataloader 
        self.attack_method = self.args.blackbox_attack
        self.targeted = self.args.targeted
        self.target_model = attack_model.cuda().eval()
        self.transfer_model = transfer_model.cuda().eval()
        
        if self.attack_method == "PI_NES":
            self.attack = PI_NES(self.target_model, self.args)
        
    def train(self):
        mseloss = nn.MSELoss()
        transfer_acc = 0
        acc = 0
        total = 0

        l0 = 0.0
        l2 = 0.0
        linf = 0.0
        for i, (img, label, index) in enumerate(self.dataloader):
            print("=================num {}/{}".format(i, len(self.dataloader)))
            img = img.cuda()
            label = label.cuda()

            _, ori_pred = self.target_model(img)
            ori_pred = torch.argmax(ori_pred, dim = 1)
            true = torch.argmax(label, dim = 1)
            mask = (ori_pred == true)
            img = img[mask]
            label = label[mask]
            ori_pred = ori_pred[mask]
            true = true[mask]
            if img.size(0) == 0:
                continue 

            total = total + img.size(0)
            if self.targeted:
                target_label = self.select_target_labels(img.size(0), label.cpu())
            else:
                target_label = label
            target_label = torch.zeros(1,10)
            target_label[:,0] = 1 
            print("---------------ori_label: ", label.cpu().numpy())    
            print("---------------target_label: ", target_label.numpy())

            adv_img = self.attack(img, target_label, self.targeted)
            target = torch.argmax(target_label.cuda(), dim = 1)
            _, attack_pred = self.target_model(adv_img)
            attack_pred = torch.argmax(attack_pred, dim = 1)
            _, transfer_pred = self.transfer_model(adv_img)
            transfer_pred = torch.argmax(transfer_pred, dim = 1)

            if self.targeted:
                mask = (attack_pred == target)
                if self.args.type == "linf":
                    if img[mask].size(0) != 0:
                        batch_linf = mseloss(img[mask].cuda(), adv_img[mask])
                        linf = linf +  batch_linf * img[mask].size(0)
                    else:
                        batch_linf = 0
                elif self.args.type == "l2":
                    batch_l2 = torch.norm(adv_img.view(img.size(0), -1) - img.view(img.size(0), -1), p = 2, dim = 1)
                    l2 = l2 + batch_l2[mask].sum()
                elif self.args.type == "l0":
                    delta = adv_img - img
                    delta = 1.0 * (delta != 0)
                    batch_l0 = torch.sum(delta.view(delta.size(0), -1), dim = 1)
                    l0 = l0 + batch_l0[mask].sum() 
                
                mask = 1.0 * (attack_pred == target)
                acc = acc + mask.sum()
                mask_transfer = 1.0 * (transfer_pred == target)
                transfer_acc = transfer_acc + mask_transfer.sum()

            else:
                mask = (attack_pred != target)
                if self.args.type == "linf":
                    if img[mask].size(0) != 0:
                        batch_linf = mseloss(img[mask].cuda(), adv_img[mask])
                        linf = linf +  batch_linf * img[mask].size(0)
                    else:
                        batch_linf = 0
                elif self.args.type == "l2":
                    batch_l2 = torch.norm(adv_img.view(img.size(0), -1) - img.view(img.size(0), -1), p = 2, dim = 1)
                    l2 = l2 + batch_l2[mask].sum()
                elif self.args.type == "l0":
                    delta = adv_img - img
                    delta = 1.0 * (delta != 0)
                    batch_l0 = torch.sum(delta.view(delta.size(0), -1), dim = 1)
                    l0 = l0 + batch_l0[mask].sum() 

                mask = 1.0 * (attack_pred != target)
                acc = acc + mask.sum()
                mask = 1.0 * (transfer_pred != target)
                transfer_acc = transfer_acc + mask.sum()
                
            for j in range(img.size(0)):
                if self.args.type == "linf":
                    if batch_linf != 0:
                        print("linf:{:.5f}, attack_pred:{}, transfer_pred:{}, target:{}, ori:{}".format(torch.sqrt(batch_linf), attack_pred[j], transfer_pred[j], target[j], ori_pred[j]))
                    else:
                        print("linf:{:.5f}, attack_pred:{}, transfer_pred:{}, target:{}, ori:{}".format(0, attack_pred[j], transfer_pred[j], target[j], ori_pred[j]))
                elif self.args.type == "l2":
                    print("l2:{:.5f}, attack_pred:{}, transfer_pred:{}, target:{}, ori:{}".format(batch_l2[j], attack_pred[j], transfer_pred[j], target[j], ori_pred[j]))
                elif self.args.type == "l0":
                    print("l0:{:.5f}, attack_pred:{}, transfer_pred:{}, target:{}, ori:{}".format(batch_l0[j], attack_pred[j], transfer_pred[j], target[j], ori_pred[j]))
            
        if self.args.type == "linf":
            if linf != 0:
                linf = torch.sqrt(linf/acc)
            print("acc:{:.5f}, transfer_acc:{:.5f}, linf:{:.5f}".format(acc/total, transfer_acc/total, linf))
        elif self.args.type == "l2":
            l2 = l2 / acc
            print("acc:{:.5f}, transfer_acc:{:.5f}, l2:{:.5f}".format(acc/total, transfer_acc/total, l2))
        elif self.args.type == "l0":
            l0 = l0 / acc 
            print("acc:{:.5f}, transfer_acc:{:.5f}, l0:{:.5f}".format(acc/total, transfer_acc/total, l0))
        return 
    
    def select_target_labels(self, batch_size, labels):
        num_classes = labels.size(1)
        target_labels = torch.zeros(batch_size, num_classes)
        for i, label in enumerate(labels):
            ori_idx = torch.argmax(label)
            target = torch.randint(low=0, high=num_classes, size = (1,))
            while target == ori_idx:
                target = torch.randint(low=0, high=num_classes, size = (1,))
            target_labels[i][target] = 1
        return target_labels  

