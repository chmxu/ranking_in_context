from losses import listMLE, approxNDCGLoss, neuralNDCG, MarginLoss
from model import RankModel
from dataset import RankTestDataset
import torch
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import json
from copy import deepcopy
import math
import argparse
import random
import itertools
from hodge_utils import hodge_rank

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--source_fold_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--fold_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n_gallery_list",
        nargs='+',
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default='dinov1',
    )
    parser.add_argument(
        "--similarity_annotation",
        type=str,
        default='dinov1',
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default='',
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default='./ckpt',
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default='/mnt/tmp/JPEGImages',
    )
    parser.add_argument(
        "--image_suffix",
        type=str,
        default='jpg',
    )



    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    return args

def get_edge_set(pred, idx_set):
    edge_set = []
    sorted_idx = pred.argsort(descending=True)
    for comb in itertools.combinations(range(len(idx_set)), 2):
        edge_set.append((idx_set[sorted_idx[comb[0]]], idx_set[sorted_idx[comb[1]]]))

    return edge_set

def get_edge_set_by_n(model, query_img, gallery_img, gallery_idx, n_gallery, only_random=False):
    edge_set = []

    if not only_random:
        gallery_img_ = gallery_img.clone()
        gallery_idx_ = deepcopy(gallery_idx)
        # sequential select
        while gallery_img_.shape[1] > 1:
            all_candidate = []
            all_candidate_idx = []

            batch_size = int(math.ceil(gallery_img_.shape[1]/n_gallery))
            gallery_img_batch = gallery_img_.reshape(batch_size, -1, *gallery_img_.size()[2:])
            query_img_batch = query_img.repeat(batch_size, 1, 1, 1)

            with torch.no_grad(), torch.autocast("cuda"):
                pred = model(query_img_batch, gallery_img_batch).squeeze(-1)
                preg_argsort = pred.argsort(-1)

            for j in range(batch_size):
                all_candidate.append(gallery_img_[:, j*n_gallery+preg_argsort[j, -1].item()])
                all_candidate_idx.append(gallery_idx_[j*n_gallery+preg_argsort[j, -1].item()])

                edge_set.extend(get_edge_set(pred[j], gallery_idx_[j*n_gallery:(j+1)*n_gallery]))

            gallery_img_ = torch.stack(all_candidate, dim=1)
            gallery_idx_ = deepcopy(all_candidate_idx)

    # random select
    # n_random = 50 if not only_random else 70
    n_random = 20 if not only_random else 30
    for _ in range(n_random):
        rand_idx = random.sample(range(len(gallery_idx)), n_gallery)
        gallery_img_ = gallery_img[:, rand_idx]
        gallery_idx_ = [gallery_idx[j] for j in rand_idx]
        pred = model(query_img, gallery_img_).view(-1)
        edge_set.extend(get_edge_set(pred, gallery_idx_))

    return edge_set

args = parse_args()

fold_idx = args.fold_idx
num_layers = 3
num_heads = 8
if 'clip-l' in args.backbone:
    hidden_dim = mlp_dim  = 1024
else:
    hidden_dim = mlp_dim = 768
n_patch = 4
n_init_gallery = 50

train_batch_size = 64
dataloader_num_workers = 8

num_epoch = 30
lr = 1e-4
weight_decay = 0

model = RankModel(num_layers, num_heads, hidden_dim, mlp_dim, n_patch, backbone=args.backbone)
model.backbone.requires_grad_(False)
model.backbone.eval()
model = model.cuda()
model.eval()

cudnn.benchmark = True

state_dict_list = []
n_gallery_list = list(map(lambda x:int(x), args.n_gallery_list))
for n_gallery in n_gallery_list:
    ckpt = torch.load('./{}/{}_fold{}_rank{}.pth'.format(args.ckpt_root, args.backbone, args.source_fold_idx, n_gallery))
    new_state_dict = {}
    for k, v in ckpt.items():
        new_state_dict[k.replace('module.', '')] = v
    state_dict_list.append(new_state_dict)


root = args.data_root
train_transforms  =  transforms.Compose([
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
try:
    train_dataset = RankTestDataset(root, args.similarity_annotation, 
                                train_transforms,
                                n_gallery=50,
                                n_init_gallery=1,
                                image_suffix=args.image_suffix)
except:
    train_dataset = RankTestDataset(root, args.similarity_annotation, 
                                train_transforms,
                                n_gallery=50,
                                n_init_gallery=1,
                                image_suffix=args.image_suffix)
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        num_workers=dataloader_num_workers,
    )

filter_result = []

all_models = []
for state_dict_idx, n_gallery in enumerate(n_gallery_list):
    model_ = deepcopy(model)
    model.load_state_dict(state_dict_list[state_dict_idx], strict=False)
    all_models.append(model_)

progress_bar = tqdm(range(len(train_loader)))
for i, (query_img, gallery_img, query_name, gallery_name) in enumerate(train_loader):
    progress_bar.update(1)

    query_img, gallery_img = query_img.to('cuda', non_blocking=True), \
            gallery_img.to('cuda', non_blocking=True),
    gallery_idx = list(range(len(gallery_name)))

    edge_set = []

    for state_dict_idx, n_gallery in enumerate(n_gallery_list):
        edge_set_ = get_edge_set_by_n(all_models[state_dict_idx], query_img, gallery_img, gallery_idx, n_gallery, only_random=n_gallery==10)
        edge_set.extend(edge_set_)

    hodge = hodge_rank(edge_set, len(gallery_idx))
    best_idx = hodge.get_global_rank().argsort()[-5:]

    output_gallery = [gallery_name[idx][0] for idx in best_idx]
    output_gallery.reverse()

    filter_result.append({"query":query_name[0], "gallery":output_gallery})

w = open('filter_result_sourcefold{}_fold{}_rank{}_{}_hodge_{}.json'.format(args.source_fold_idx, fold_idx, '_'.join(args.n_gallery_list), args.backbone, args.suffix), 'w')
json.dump(filter_result, w)



