import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_loss(logits_s, logits_w, k=1, t=0.95):
    # 1. 处理数据
    softmax_pred = torch.softmax(logits_s, axis=-1)  # s 计算推理概率
    pseudo_label = torch.softmax(logits_w, axis=-1)  # w 伪标签
    # 2. get mask
    max_probs_s, targets_u_s = torch.max(softmax_pred, dim=-1)
    max_probs_w, targets_u_w = torch.max(pseudo_label, dim=-1)
    mask = ((targets_u_s == targets_u_w) & max_probs_s.ge(t-0.1)) & max_probs_w.ge(t-0.1)
    mask = (mask | max_probs_w.ge(t)).float() # 大于阈值并且强弱标签预测一致的图像
    # 3.topk loss
    topk = torch.topk(pseudo_label, k)[1]  # [n, k] # 计算topk的pseudo_label中概率最高的k个元素的索引
    mask_k = torch.scatter(torch.ones_like(pseudo_label), 1, topk, 0)  # 创建一个mask,topk位置设为0,其它是1
    mask_k = torch.where((mask_k==1)&(softmax_pred>t**3), torch.zeros_like(mask_k), mask_k) 
    # 非top1的趋向于0
    loss_non1 = ((-torch.log(1-softmax_pred+1e-10) * mask_k).sum(axis=1) * mask).mean()
    # 4. kl loss
    k = cal_topK(logits_s.detach(), logits_w.detach(), topk=(1,logits_s.shape[-1]))
    topk = torch.topk(pseudo_label, k)[1]
    mask_k = torch.scatter(torch.zeros_like(pseudo_label), 1, topk, 1)  # 前k个位置是1
    # mask_k = (mask_k*(1-mask)[...,None])  # 获得阈值之下的前k个位置为1

    # # 软标签
    # targets_u = pseudo_label.detach()
    # targets_u = targets_u ** 2
    # targets_u = (targets_u / targets_u.sum(dim=1, keepdim=True))
    targets_u = pseudo_label.detach()
    
    loss_topk = -targets_u*(targets_u*torch.log(softmax_pred+1e-10) + (1-targets_u)*torch.log(1-softmax_pred+1e-10))
    loss_topk = torch.sum(loss_topk * mask_k, dim=1).mean()
    return loss_non1,  loss_topk



def entropy_loss1(logits_s, logits_w, k=1, t=0.95):
    # 1. 处理数据
    softmax_pred = torch.softmax(logits_s, axis=-1)  # s 计算推理概率
    pseudo_label = torch.softmax(logits_w, axis=-1)  # w 伪标签
    # 2. get mask
    max_probs_s, targets_u_s = torch.max(softmax_pred, dim=-1)
    max_probs_w, targets_u_w = torch.max(pseudo_label, dim=-1)
    mask = ((targets_u_s == targets_u_w) & max_probs_s.ge(t-0.1)) & max_probs_w.ge(t-0.1)
    mask = (mask | max_probs_w.ge(t)).float() # 大于阈值并且强弱标签预测一致的图像
    # 3.topk loss
    topk = torch.topk(pseudo_label, k)[1]  # [n, k] # 计算topk的pseudo_label中概率最高的k个元素的索引
    mask_k = torch.scatter(torch.ones_like(pseudo_label), 1, topk, 0)  # 创建一个mask,topk位置设为0,其它是1
    mask_k = torch.where((mask_k==1)&(softmax_pred>t**3), torch.zeros_like(mask_k), mask_k) 
    # 非top1的趋向于0
    loss_non1 = ((-torch.log(1-softmax_pred+1e-10) * mask_k).sum(axis=1) * mask).mean()
    return loss_non1






def getLossTopk(logits_s, logits_w, k=1, t=0.95):
    # 1. 处理数据
    softmax_pred = torch.softmax(logits_s, axis=-1)  # s 计算推理概率
    pseudo_label = torch.softmax(logits_w, axis=-1)  # w 伪标签
    # 2. get mask
    max_probs_s, targets_u_s = torch.max(softmax_pred, dim=-1)
    max_probs_w, targets_u_w = torch.max(pseudo_label, dim=-1)
    mask = ((targets_u_s == targets_u_w) & max_probs_w.ge(t-0.05)) & max_probs_s.ge(t-0.05)
    mask = (mask | max_probs_s.ge(t)).float() # 大于阈值并且强弱标签预测一致的图像
    # 3.topk loss
    ###### 1. 计算大于阈值并且强弱标签预测一致的图像，让其cls-1个趋向于0
    topk = torch.topk(pseudo_label, k)[1]  # [n, k] # 计算topk的pseudo_label中概率最高的k个元素的索引
    # 函数用于将一个张量按照指定的索引进行散射（scatter），即将指定位置的值替换为给定的值, 每个预测最大的k个位置设为0
    mask_k = torch.scatter(torch.ones_like(pseudo_label), 1, topk, 0)  # 创建一个mask,topk位置设为0,其它是1
    # 设置不是前k个，但是强增强预测概率比较大的不计算损失
    mask_k = torch.where((mask_k==1)&(softmax_pred>t**3), torch.zeros_like(mask_k), mask_k) 
    # 计算大于阈值并且强弱标签预测一致的图像，让其cls-1个趋向于0
    loss_topk = ((-torch.log(1-softmax_pred+1e-10) * mask_k).sum(axis=1) * mask).mean()
    # 4. kl loss
    topk = torch.topk(pseudo_label, 5)[1]
    mask_k = torch.scatter(torch.zeros_like(pseudo_label), 1, topk, 1)  # 前k个位置是1
    mask_k = (mask_k*(1-mask)[...,None])  # 获得阈值之下的前k个位置为1
    loss_kl = F.kl_div(F.log_softmax(logits_s / 0.5, dim=-1), F.softmax(logits_w / 0.5, dim=-1), reduction='none')
    loss_kl = torch.sum(loss_kl * mask_k, dim=1).mean()
    return loss_topk + loss_kl

def ourloss1(logits_s, logits_w, k=1, t=0.95):
    # 1. 处理数据
    softmax_pred = torch.softmax(logits_s, axis=-1)  # s 计算推理概率
    pseudo_label = torch.softmax(logits_w, axis=-1)  # w 伪标签
    # 2. get mask
    max_probs_s, targets_u_s = torch.max(softmax_pred, dim=-1)
    max_probs_w, targets_u_w = torch.max(pseudo_label, dim=-1)
    
    mask = ((targets_u_s == targets_u_w) & max_probs_w.ge(t-0.05)) & max_probs_s.ge(t-0.05)
    mask = (mask | max_probs_w.ge(t)).float() # 大于阈值并且强弱标签预测一致的图像
    
    # 3.topk loss
    ###### 1. 计算大于阈值并且强弱标签预测一致的图像，让其后面cls-1个趋向于0
    topk = torch.topk(pseudo_label, k)[1]  # [n, k] # 计算topk的pseudo_label中概率最高的k个元素的索引
    # 函数用于将一个张量按照指定的索引进行散射（scatter），即将指定位置的值替换为给定的值, 每个预测最大的k个位置设为0
    mask_k = torch.scatter(torch.ones_like(pseudo_label), 1, topk, 0)  # 创建一个mask,topk位置设为0,其它是1
    # 设置不是前k个，但是强增强预测概率比较大的不计算损失
    mask_k = torch.where((mask_k==1)&(softmax_pred>t**3), torch.zeros_like(mask_k), mask_k) 
    # 计算大于阈值并且强弱标签预测一致的图像，让其cls-1个趋向于0
    loss_topk = ((-torch.log(1-softmax_pred+1e-10) * mask_k).sum(axis=1) * mask).mean()
    
    return loss_topk

def ourloss3(logits_s, logits_w, simMatrix,t=0.95):

    logits_w = logits_w.detach()
    # 1.处理数据
    logits_w = torch.softmax(logits_w.detach(), axis=-1)
    max_probs, pseudo_l = torch.max(logits_w, -1)
    mask = max_probs.lt(t) # gao于阈值
    logits_s = torch.softmax(logits_s, 1) # softmax
    
    # 计算损失
    logits_w, max_probs, pseudo_l = logits_w[mask], max_probs[mask], pseudo_l[mask]
    logits_s_ = logits_s[mask]

    sim_ratio = max_probs.view(-1,1) - logits_w
    isthan = (sim_ratio > simMatrix[pseudo_l]).float()  # 计算是否超过安全置信度
    isthan = torch.where((isthan==1)&(logits_s_>t**3), torch.zeros_like(isthan), isthan)  # 如果强增强的太大，设为0
    loss = 0
    if isthan.sum() > 0:
        loss = (((-torch.log(1-logits_s_+1e-10)) * isthan.float()).sum()) / isthan.sum()
    return loss, isthan.sum() / (isthan.shape[0] * isthan.shape[-1])  # 返回损失和有效的比例

def klloss(logits, targets, mask):
    loss = F.kl_div(F.log_softmax(logits / 0.5, dim=-1), F.softmax(targets / 0.5, dim=-1), reduction='none')
    loss = torch.sum(loss * (1.0 - mask).unsqueeze(dim=-1).repeat(1, logits.shape[1]), dim=1)
    return loss.mean()



def Calculate_null_loss(logits_s, mask, pseudo_label=None):

    logits_s = torch.softmax(logits_s, -1)
    if pseudo_label is None: 
        loss = (((-torch.log(1-logits_s+1e-10)) * mask).sum(-1)).mean()
    else:
        loss = (((-torch.log(1-logits_s+1e-10)) * mask * (1-pseudo_label)).sum(-1)).mean()
    return loss

####################################
#处理混淆类的损失
####################################

def CosineLoss(x1, x2, targets, ):
    return F.cosine_embedding_loss(x1, x2, targets, margin=0.5, reduction='mean')

def MultiMarginLoss(logits_s, targets, simMatrix, mask):
    # # loss4  提升置信度，区分类别
    # mask是筛选到伪标签的， simMatrix
    # loss_instance = (F.multi_margin_loss(logits_u_s, targets_u,reduction='none') * mask).mean()
    pass




####################################
####################################

class SupConLoss(nn.Module): 

    def __init__(self, temperature=0.06,): # after hyperparam optimization, temperature of 0.06 gave the best ICBHI score
        super().__init__()
        #temperature对loss进行缩放
        self.temperature = temperature
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, projection1, projection2, labels=None):
        # 对特征归一化
        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        #特征拼接
        features = torch.cat([projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1)
        #batch_size
        batch_size = features.shape[0]
        #lables和mask不可以同时存在，labels存在则为有监督对比学习
        #mask是将labels中属于同一类的变成1例如torch.tensor([1,1,3]) -> mask = tensor([[1., 1., 0.],[1., 1., 0.],[0., 0., 1.]])
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.device)

        #两个数据增强
        contrast_count = features.shape[1]
        #拆分tensor，每个张量为一个列表再次拼接得到(2 * batch_size, feature_size)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #计算了所有 features 两两之间的余弦相似度(内积)得到（2 * batch_size，2 * batch_size]）
        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)
        #记录了每一行的最大值。 也就是与每个样本有最大多少的相似度。(最大不超过1)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        #广播相减,(由于后面要进行指数操作,因此全负数为了数值稳定性。这个计算在loge方运算中不改变最后的计算值)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability
        #为了完全对齐相应logits的形状 得到(2 * batch_size, 2 * batch_size)
        mask = mask.repeat(contrast_count, contrast_count)
        #logits_mask = （2 * batch_size, 2 * batch_size）
        #其中全矩阵值为1， 而对角线值为0。logits_mask 用于保证每个feature不与自己本身进行对比
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device), 0)
        # mask = mask * logits_mask之后， mask中所有非自己的同类别样本上都是1。
        mask = mask * logits_mask

        #所有的正样本指数运算后的值,自身相乘为0
        exp_logits = torch.exp(logits) * logits_mask
        #以下为公式
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        
        return loss

####################################
# 通过不同类别的K值计算loss
####################################
# 针对与无标签数据, pseudo_label=softmax(w)
# 需要传入没有伪标签的数据
def k_null_loss(logits_s, pseudo_label, mask_select, CK, simMatrix):
    B, C = pseudo_label.shape
    logits_s = logits_s.softmax(-1)

    sorted_pseudo_label, sorted_idx = pseudo_label.topk(C, dim=-1, sorted=True) # 进行排序
    sorted_logits_s_all = logits_s.gather(dim=-1, index=sorted_idx) # 全部数据
    P_labels_all = sorted_idx[:,0] # pse labels

    sorted_logits_s = sorted_logits_s_all[mask_select] # 被筛选的数据
    P_labels = P_labels_all[mask_select]  # 获取伪标签
    max_value = sorted_pseudo_label[mask_select][:,0] # 获取最大值
    sorted_idx_select = sorted_idx[mask_select]
    # 对类别数进行循环
    loss_null = 0
    loss_ratio = 0
    for c in range(C):
        k = int(CK[c]) # 获取当前类别的k值
        if k == C or k < 1: continue  # K等于类别数,或者效果特别的好，不做loss
        
        # 1. loss_null 使用全部的数据
        mask = P_labels_all==c   # 获取当前类别的强增强数据
        if not mask.sum() > 0:
            continue
        sorted_logits_s_k_c = sorted_logits_s_all[mask][:,k:]  # 获取后K个数据 
        loss_null += (-torch.log(1-sorted_logits_s_k_c+1e-10)).sum(-1).mean()  # 后面K个，趋向于0

        # loss_ratio,按照相似的比例进行损失,仅对大于阈值的标签
        mask = P_labels==c   # 获取当前类别的强增强数据
        if not mask.sum() > 0:
            continue
        sorted_cls = sorted_idx_select[mask]  # [N, num_class]属于当前类别的样本，其他预测类别的排序
        sim = simMatrix[c].repeat(mask.sum()).view(-1, C)  # 【N,num_class】属于当前类别的相似矩阵
        sim = sim.gather(dim=-1, index=sorted_cls)  # [N, num_class]每个样本不同的相似
        sim = sim[:, 1:k]  #[N, k-1]
        sim_ratio = (1-sim) / (1-sim).sum(-1,keepdim=True)  # [N, k-1]求比例，小的相似度应该有最大的越策概率
        sorted_logits_s_1_k = sorted_logits_s[mask][:,1:k]  # [N, k-1]获取中间的k-1个【N， k-1】
        p_labels_ = sorted_logits_s_1_k.detach().sum(-1, keepdim=True) * sim_ratio  # 构建临时标签
        loss_ = -(p_labels_*torch.log(sorted_logits_s_1_k+1e-10)+(1-p_labels_)*torch.log(1-sorted_logits_s_1_k+1e-10))
        w = max_value[mask]   # 权重
        loss_ratio += (loss_.sum(-1) * w).sum() / (mask.sum() * (k - 1))
        
    return loss_null / C + loss_ratio / C



####################################
####################################