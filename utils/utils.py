
import torch
from torchvision import transforms
import random
import numpy as np

#####################################
#计算相似度
#####################################
class SimilarMatrix(object):

    def __init__(self, num_classes, device, simMatrix=None, k_=None, t=0.9, alph=0.999):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        if not simMatrix is None:
            self.sm = torch.from_numpy(simMatrix).to(device)
        else:
            self.sm = torch.ones(num_classes, num_classes).to(device)
                
        if not k_ is None:
            self.k_ = torch.from_numpy(k_).to(device)
        else:
            self.k_ = torch.ones(num_classes) * num_classes  # 计算得到的

        self.t = t
        self.alph = alph
        self.eyes = torch.eye(num_classes, num_classes).to(device).to(bool)
        self.k = torch.ceil(self.k_)  # 最终的
    
    @torch.no_grad()
    def updata(self, logits_w, pseudeLabels=None):
        '''
        logits: weak img inference 
        updata sm
        '''
        if pseudeLabels is None:  # 如果传入的是没有经过处理的弱增强推理结果
            logits = logits_w.detach()
            # 筛选数据
            logits = torch.softmax(logits, -1)
            max_probs, pseudeLabels = torch.max(logits,-1)
            mask = max_probs.lt(self.t)
            logits = logits[mask]
            pseudeLabels = pseudeLabels[mask]
        else:  # 传入有伪标签和处理的结果
            logits = logits_w
        # 循环更新矩阵
        temp_sm = torch.zeros_like(self.sm)
        for i in range(self.num_classes):
            mask_ = pseudeLabels==i
            if mask_.sum() > 0:
                data = logits[mask_]  # 获取预测为i类别的数据
                # 1.相似矩阵的处理
                temp = data.max(-1)[0].view(-1,1) - data
                if mask_.sum() == 1:
                    temp_sm[i] = temp.mean(0)
                else:
                    temp_sm[i] = temp.mean(0) + temp.std(0)

                # 2. k值处理
                data = data.mean(0)
                sortData = data.sort(dim=-1,descending=True)[0]  # 排序
                for k_ in range(1, self.num_classes+1):
                    # if torch.all(sortData[:, :k_].sum(-1) >= 0.95):
                    if sortData[:k_].sum() > 0.95:
                        self.k_[i] = self.k_[i] * self.alph + (1-self.alph) * (k_ - 0.1)
                        break
            else:   
                temp_sm[i] = self.sm[i]
        self.sm = self.sm * self.alph + (1-self.alph) * temp_sm
        self.sm = torch.where(self.eyes, torch.ones_like(self.sm), self.sm)
        self.k = torch.ceil(self.k_)

