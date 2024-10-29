from losses import listMLE, approxNDCGLoss, neuralNDCG, MarginLoss
from model import RankModel
from dataset import RankDataset
import torch
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import argparse
from accelerate import Accelerator
import os
from accelerate.utils import set_seed


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--fold_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--n_gallery",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default='dinov1',
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--pair_file",
        type=str,
        default=None,
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    return args
# set_seed(42)
args = parse_args()
env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
if env_local_rank != -1 and env_local_rank != args.local_rank:
    args.local_rank = env_local_rank

accelerator = Accelerator()

fold_idx = args.fold_idx
num_layers = 3
num_heads = 8

if 'clip-l' in args.backbone:
    hidden_dim = mlp_dim  = 1024
else:
    hidden_dim = mlp_dim = 768

n_gallery = args.n_gallery
n_patch = 4
n_init_gallery = 50

train_batch_size = args.train_batch_size
dataloader_num_workers = 8

num_epoch = 80
lr = args.lr
weight_decay = 0

model = RankModel(num_layers, num_heads, hidden_dim, mlp_dim, n_patch, backbone=args.backbone)
model.backbone.requires_grad_(False)

try:
    state_dict = torch.load('/apdcephfs_cq10/share_1275017/chengmingxu/model_weights/dinov2_vitb14_pretrain.pth')
except:
    state_dict = torch.load('/apdcephfs_cq8/share_2992679/private/chengmingxu/model_weights/dinov2_vitb14_pretrain.pth')
model.backbone.load_state_dict(state_dict)
# print(msg)

params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)


root = '/mnt/tmp/JPEGImages/'
train_transforms  =  transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

train_dataset = RankDataset(root, '/apdcephfs/private_chengmingxu/InContext_NIPS2024/VPR/seg/visual_prompt_retrieval/output_seg_images/output_vit-laion2b-clip_trn_folder{}_seed0/annotation.json'.format(fold_idx), 
                            train_transforms,
                            n_gallery=n_gallery,
                            n_init_gallery=n_init_gallery,
                            fold_idx=fold_idx,
                            pair_file=args.pair_file)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
    )

# model.train()

train_loader, model, optimizer = accelerator.prepare(
    train_loader, model, optimizer
)

progress_bar = tqdm(
    range(0, num_epoch*len(train_loader)),
    desc="Steps",
    disable=not accelerator.is_local_main_process,
)

for e in range(num_epoch):
    try:
        model.module.backbone.eval()
    except:
        model.backbone.eval()
        
    total_rank_loss = 0
    total_reg_loss = 0
    total_pair_loss = 0
    for i, (query_img, gallery_img, label, score, rank_labels) in enumerate(train_loader):
        progress_bar.update(1)
        pred = model(query_img, gallery_img)

        pair_rank_loss = MarginLoss(pred[:, :, 0], score.float())
        rank_loss = neuralNDCG(pred[:, :, 0], label.float())

        loss = 0.5 * rank_loss + pair_rank_loss

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_rank_loss = accelerator.gather(rank_loss.repeat(train_batch_size)).mean().item()
        avg_pair_loss = accelerator.gather(pair_rank_loss.repeat(train_batch_size)).mean().item()

        total_rank_loss += avg_rank_loss
        total_pair_loss += avg_pair_loss

        logs = {"Rank Loss": total_rank_loss/(i+1), "Pair Loss": total_pair_loss/(i+1)}
        progress_bar.set_postfix(**logs)
    if accelerator.is_main_process:
        print("Epoch {}, Pair Loss {}, Rank Loss {}".format(e, total_pair_loss/(i+1), total_rank_loss/(i+1)))

if accelerator.is_main_process:
    torch.save(model.state_dict(), './ckpt/{}_fold{}_rank{}_debug.pth'.format(args.backbone, fold_idx, n_gallery))
    # print('Epoch {} AvgRankLoss {} AvgRegLoss {}'.format(e, total_rank_loss/len(train_loader), total_reg_loss/len(train_loader)))