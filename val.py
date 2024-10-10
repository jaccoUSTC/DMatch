import argparse
import contextlib
import logging
import math
import os
import random
import shutil
import time


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from dataset.cifar_mix import DATASET_GETTERS
from utils import AverageMeter, accuracy
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)
best_acc = 0

class SimilarMatrix(object):

    def __init__(self, num_classes, device, simMatrix=None, k_=None, t=0.9, alph=0.99):
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
        # 循环更新矩阵,获取相似矩阵，和不同类别的K
        temp_sm = torch.zeros_like(self.sm)
        for i in range(self.num_classes):
            mask_ = pseudeLabels==i
            if mask_.sum() > 0:
                data = logits[mask_]  # 获取预测为i类别的数据
                # 1.相似矩阵的处理
                temp = data.max(-1)[0].view(-1,1) - data
                temp_sm[i] = temp.mean(0)
                
                # # 2. k值处理
                data = data.mean(0)
                sortData = data.sort(dim=-1,descending=True)[0]  # 排序
                for k_ in range(1, self.num_classes+1):
                    if sortData[:k_].sum() > 0.99:
                        self.k_[i] = self.k_[i] * self.alph + (1-self.alph) * (k_ - 0.1)
                        break
            else:   
                temp_sm[i] = self.sm[i]

        self.sm = self.sm * self.alph + (1-self.alph) * temp_sm
        self.sm = torch.where(self.eyes, torch.ones_like(self.sm), self.sm)
        self.k = torch.ceil(self.k_)


def k_null_loss(logits_s, pseudo_label, CK, select_mask):
    _, C = pseudo_label.shape
    # logits_s,pseudo_label = logits_s[select_mask], pseudo_label[select_mask]
    logits_s = logits_s.softmax(-1)

    weak_score, sorted_idx = pseudo_label.topk(C, dim=-1, sorted=True)  # 进行排序
    sorted_logits_s_all = logits_s.gather(dim=-1, index=sorted_idx) # 全部数据
    P_labels_all = sorted_idx[:,0] # pse labels

    soft_lables = torch.zeros_like(logits_s)  # 全为0
    mask_nlp = torch.zeros_like(logits_s) 
    
    for c in range(C):
        mask_ = P_labels_all==c
        soft_lables[mask_,1:int(CK[c])] = (1-weak_score[mask_,0].mean(0)) / (C-1)  # 混淆类平均
        mask_nlp[mask_,int(CK[c]):] = 1
    
    
    # loss-1
    mask_nlp = torch.where((mask_nlp==1)&(sorted_logits_s_all>0.5), torch.zeros_like(mask_nlp), mask_nlp)
    loss_nlp = (-torch.log(1-sorted_logits_s_all+1e-10) * mask_nlp).sum(-1).mean()

    # loss2    
    mask = torch.where(soft_lables>0, 1, 0) # mask
    loss_topk = -(soft_lables*torch.log(sorted_logits_s_all+1e-10) + (1-soft_lables)*torch.log(1-sorted_logits_s_all+1e-10))
    loss_topk = (loss_topk * mask).sum() / (mask.sum()+1e-10)
    return loss_topk + loss_nlp


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='results/others',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", default=True,action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    parser.add_argument('--Ablation_study',type=bool, default=False, help="是否消融")
    parser.add_argument('--dropout_ratio', default=-1, type=float)
    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnetCa as models
            # import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=args.dropout_ratio,
                                            num_classes=args.num_classes)
            
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)
    print(model)
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    args.simMatrix = None
    args.simMatrix_k_ = None
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.simMatrix = checkpoint['simMatrix']
        args.simMatrix_k_ = checkpoint['simMatrix_k_']
    
    # if args.amp:
    #     from apex import amp
    #     model, optimizer = amp.initialize(
    #         model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    scaler = GradScaler()
    amp_cm = autocast if args.amp else contextlib.suppress
    simMatrix = SimilarMatrix(args.num_classes, args.device, simMatrix=args.simMatrix)


    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x, id_l = labeled_iter.next()
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x, id_l = labeled_iter.next()
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), labels_u, id_u = unlabeled_iter.next()
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), labels_u, id_u = unlabeled_iter.next()
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            
            
            with amp_cm():
                inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
                targets_x = targets_x.to(args.device)
                logits = model(inputs)
                logits = de_interleave(logits, 2*args.mu+1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits
                
                # 1.有标签loss计算
                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                # 2.获取为标签，处理数据
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()

                # 3.无标签loss计算
                Lu_temp = F.cross_entropy(logits_u_s, targets_u,reduction='none') # [N]
                Lu =  (Lu_temp* mask).mean()

                # 4. k_null loss calculate
                L_null = 0
                if 5 <= epoch:
                    L_null = k_null_loss(logits_u_s, pseudo_label,simMatrix.k, select_mask=mask.bool())

                loss = Lx + args.lambda_u * Lu + L_null

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            # 更新数据库
            simMatrix.updata(pseudo_label[~mask.bool()], targets_u[~mask.bool()])  # 更新
            unlabeled_iter._dataset.updata_data(idx=id_u, current_loss=Lu_temp, current_cls=targets_u, current_sim=simMatrix.sm.argmin(-1), epoch=epoch)

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            # print out
            if batch_idx%50==0:
                P_acc_ = 1 if mask.sum() == 0 else (targets_u[mask.to(bool)] == labels_u[mask.to(bool)].to(args.device)).float().mean().item()
                # print(f'simMatrix.k{simMatrix.k_}')

                print(f"Train Epoch: {epoch}/{args.epochs:4}. Iter: {batch_idx + 1:4}/{args.eval_step:4}. LR: {scheduler.get_last_lr()[0]:.4f}. Loss: {losses.avg:.4f}. Loss_sup: {losses_x.avg:.4f}. Loss_unsup: {losses_u.avg:.4f}. L_null:{L_null:.4f}. Mask: {mask_probs.avg:.4f}. P_acc:{P_acc_:.4f}. mean_k:{simMatrix.k_.mean():.4f}")

        print(simMatrix.k_.to('cpu').numpy(), simMatrix.k.to('cpu').numpy())


        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'simMatrix': simMatrix.sm.to('cpu').numpy(),
                'simMatrix_k_': simMatrix.k_.to('cpu').numpy(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            
            loss = F.cross_entropy(outputs, targets)

            prec1, prec2, prec5 = accuracy(outputs, targets, topk=(1, 2, 5))
            
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top2.update(prec2.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx%10==0:
                print(f"Test Iter: {batch_idx + 1:4}/{len(test_loader):4}. Loss: {losses.avg:.4f}. top1: {top1.avg:.2f}. top2: {top2.avg:.2f}. top5: {top5.avg:.2f}. ")

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-2 acc: {:.2f}".format(top2.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
