import torch
import json
from PIL import Image
import numpy as np
import os
import random
from tqdm import tqdm
import sys

class RankDataset(torch.utils.data.Dataset):
        
    def __init__(self, root, annotation_file, transforms, n_gallery=10, n_init_gallery=12, fold_idx=0, pair_file=None, suffix='seg'):
        self.root = root
        self.transforms = transforms
        self.n_gallery = n_gallery
        self.n_init_gallery = n_init_gallery

        with open(annotation_file, encoding="utf-8") as f:
            self.data = json.load(f)

        all_data = []
        all_data_dummy = []

        if pair_file is None:
            for line in tqdm(self.data):
                query_image, gallery_image, score = line['query'], line['gallery'], line['score']
                # query_image_ = Image.open(os.path.join(self.root, '{}.jpg'.format(query_image))).convert('RGB')
                pair_list = list(zip(gallery_image, score))

                for i in range(n_init_gallery):
                    gallery_image_sub, score_sub = zip(*random.sample(pair_list, n_gallery))
                    # all_data.append({"query":query_image_, "gallery":gallery_image_sub, "score":score_sub})
                    all_data_dummy.append({"query":query_image, "gallery":gallery_image_sub, "score":score_sub})

            with open('./tmp_annotations/pair_files_{}_{}_{}.json'.format(n_gallery, fold_idx, suffix), 'w') as w:
                json.dump(all_data_dummy, w)
            self.data = all_data_dummy
        else:
            print('Loading pre-processed data pairs...')
            all_data = json.load(open(pair_file))

            self.data = all_data
   
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        query_image, gallery_image, score = self.data[idx]['query'], self.data[idx]['gallery'], self.data[idx]['score']
        try:
            query_image = Image.open(os.path.join(self.root, '{}.jpg'.format(query_image))).convert('RGB')
        except:
            query_image = Image.open(os.path.join(self.root, '{}.JPEG'.format(query_image))).convert('RGB')
        if self.transforms is not None:
            query_image = self.transforms(query_image)

        gallery_idx = np.random.permutation(len(gallery_image))[:self.n_gallery]

        gallery_image = [gallery_image[i] for i in gallery_idx]
        score = [float(score[i]) for i in gallery_idx]

        label = np.argsort(score)
        score = np.array(score)

        gallery_image_list = []
        rank_labels = []
        for i, item in enumerate(gallery_image):
            try:
                gallery_img = Image.open(os.path.join(self.root, '{}.jpg'.format(item))).convert('RGB')
            except:
                gallery_img = Image.open(os.path.join(self.root, '{}.JPEG'.format(item))).convert('RGB')
            if self.transforms is not None:
                gallery_img = self.transforms(gallery_img)
            gallery_image_list.append(gallery_img)
            if i < self.n_gallery-1:
                rank_label = 1 if score[i]>score[i+1] else -1
                rank_labels.append(rank_label)

        gallery_image_list = torch.stack(gallery_image_list, 0)
        rank_labels = torch.tensor(rank_labels)

        return query_image, gallery_image_list, label, score, rank_labels
    
class RankTestDataset(torch.utils.data.Dataset):
        
    def __init__(self, root, annotation_file, transforms, n_gallery=10, n_init_gallery=12, image_suffix='jpg'):
        self.root = root
        self.transforms = transforms
        self.n_gallery = n_gallery
        self.n_init_gallery = n_init_gallery

        with open(annotation_file, encoding="utf-8") as f:
            self.data = json.load(f)

        all_data = []
        for query_image,gallery_image  in self.data.items():
        #     query_image_ = Image.open(os.path.join(self.root, '{}.jpg'.format(query_image))).convert('RGB')

            all_data.append({"query":query_image, "gallery":gallery_image})

        if len(all_data) > 1000:
            np.random.seed(0)
            all_idx = np.random.choice(len(all_data), 1000, replace=False)
            all_data = [all_data[i] for i in all_idx]

        self.data = all_data
        self.image_suffix = image_suffix
        # import pdb;pdb.set_trace()
   
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        query_image_name, gallery_image_name = self.data[idx]['query'], self.data[idx]['gallery']
        query_image = Image.open(os.path.join(self.root, '{}.{}'.format(query_image_name.replace('.', ''), self.image_suffix))).convert('RGB')
        if self.transforms is not None:
            query_image = self.transforms(query_image)

        gallery_image_list = []
        for i, item in enumerate(gallery_image_name):
            try:
                gallery_img = Image.open(os.path.join(self.root, '{}.{}'.format(item.replace('.', ''), self.image_suffix))).convert('RGB')
            except:
                gallery_img = Image.open(os.path.join(self.root, '{}.jpg'.format(item.split(' ')[0].replace('.', ''), self.image_suffix))).convert('RGB')
            if self.transforms is not None:
                gallery_img = self.transforms(gallery_img)
            gallery_image_list.append(gallery_img)

        gallery_image_list = torch.stack(gallery_image_list, 0)

        return query_image, gallery_image_list, query_image_name, gallery_image_name
