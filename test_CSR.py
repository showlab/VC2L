import logging
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from training.render_img_text_np import get_pp_render_text
from PIL import Image
from contextlib import suppress
from tqdm import tqdm
from open_clip import create_model_and_transforms
import warnings
warnings.filterwarnings('ignore')

DATASET_ROOT = "path_to_dataset"
ckpt_list = ["path_to_ckpt"]
class SlidePairDataset(Dataset):
    def __init__(
        self, 
        transforms,
        ann_file = "ppt_val.txt",
        data_root = DATASET_ROOT
    ):
        self.pre_data_list = []
        self.nxt_data_list = []
        with open(ann_file, 'r') as f:
            for line in f:
                pre_data, nxt_data = line.strip().split(',')
                self.pre_data_list.append(data_root+pre_data)
                self.nxt_data_list.append(data_root+nxt_data)
        self.transforms = transforms
    

    def __len__(self):
        return len(self.pre_data_list)

    def __getitem__(self, idx):
        pre_data = self.pre_data_list[idx]
        nxt_data = self.nxt_data_list[idx]
        pre_data = Image.open(pre_data)
        pre_data = self.transforms(pre_data)
        nxt_data = Image.open(nxt_data)
        nxt_data = self.transforms(nxt_data)
        return pre_data, nxt_data

def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def evaluate(model, dataloader, device, amp=True, recall_k_list=[1,5,10]):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------
    
    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use
    
    Returns
    -------
    
    dict of retrieval metrics
    """
    # list of batch of images embedding
    batch_pre_data_emb_list = []
    # list of batch of text embedding
    batch_nxt_data_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    pair_index = []
    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress
    for batch_pre_data, batch_nxt_data, inds in tqdm(dataloader):
        batch_pre_data = batch_pre_data.to(device)
        batch_nxt_data = batch_nxt_data.to(device)
        batch_pair_index = [ind for ind in inds]
        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            batch_pre_data_emb = F.normalize(model.encode_image(batch_pre_data), dim=-1)
            batch_nxt_data_emb = F.normalize(model.encode_image(batch_nxt_data), dim=-1)

        batch_pre_data_emb_list.append(batch_pre_data_emb.cpu())
        batch_nxt_data_emb_list.append(batch_nxt_data_emb.cpu())
        pair_index.extend(batch_pair_index)
        
    batch_size = len(batch_pre_data_emb_list[0])

    # concatenate all embeddings
    pre_data_emb = torch.cat(batch_pre_data_emb_list)
    nxt_data_emb = torch.cat(batch_nxt_data_emb_list)
    # get the score for each text and image pair
    scores = nxt_data_emb @ pre_data_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), pair_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"pre2nxt_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, 1024, device, k=recall_k)>0).float().mean().item()
        metrics[f"nxt2pre_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, 1024, device, k=recall_k)>0).float().mean().item()

    return metrics

def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)



def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None):
    model, _, transform = create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)
    model = model.cuda()
    return model, transform



for ckpt in ckpt_list:
    model, transform = load_open_clip("ViT-B-16-448", ckpt)
    dataset = SlidePairDataset(transform)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=128, 
                    shuffle=False, num_workers=8, 
                )
    print("num of pairs: ", len(dataset))
    print(ckpt, evaluate(model, dataloader, "cuda"))
