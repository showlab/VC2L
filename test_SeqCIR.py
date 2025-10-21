import ast
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import Value
import glob
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from training.render_img_text_np import get_pp_render_text
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from pathlib import Path
import orjson
from contextlib import suppress
import torch.nn.functional as F
from tqdm import tqdm
OBLICES_PATH = "path_to_oblices_ann"
OBLICES_SEEK_MAP_PATH = "path_to_oblices_ann_seek_map"
ckpt_list = ["path_to_ckpt"]

def get_val_data():
    random.seed(42)
    valid_data = []
    path_mmc4 = Path(OBLICES_PATH)
    seek_map_mmc4 = Path(OBLICES_SEEK_MAP_PATH)
    seek_map_mmc4 = orjson.loads(seek_map_mmc4.open().read())
    dataset_file_mmc4 = None
    for idx in range(len(seek_map_mmc4)):
        position = seek_map_mmc4[idx]
        if dataset_file_mmc4 is None:
            dataset_file_mmc4 = path_mmc4.open()
        dataset_file_mmc4.seek(position)
        line = dataset_file_mmc4.readline()
        data = orjson.loads(line)
        valid_data.append(data)

    for doc_id, data in enumerate(valid_data):
        valid_data[doc_id]['items'] = []
        for text_id, (text, image) in enumerate(zip(data["text"], data["img"])):
            if text_id + 1 < len(data["text"]):
                is_end = 0
            else:
                is_end = 1
            valid_data[doc_id]['items'].append(
                {
                    "text": text,
                    "image": image,
                    "doc_id": doc_id,
                    "text_id": text_id,
                    "is_end": is_end,
                    'uniid': doc_id * 10000 + text_id * 10 + is_end,
                    "random_idx": random.sample([0, 1, 2, 3], 1)[0]
                }
            )
    return valid_data


class SeqDocDataset(Dataset):

    def __init__(self, valid_data, transforms, skip_img=False, skip_txt=False, fliter_len=0):
        self.valid_data = valid_data
        self.skip_img = skip_img
        self.skip_txt = skip_txt
        self.tokenize = get_pp_render_text(max_chars=1100)
        self.items = []
        if fliter_len:
            for doc_id, data in enumerate(valid_data):
                if len(data['items']) < fliter_len:
                    continue
                for item_id, item in enumerate(data['items']):
                    if item_id + 1 < fliter_len:
                        self.items.append(item)
        else:
            for doc_id, data in enumerate(valid_data):
                for item_id, item in enumerate(data['items']):
                    self.items.append(item)
        
        self.length = len(self.items)
        self.transforms = transforms
        logging.debug('Done loading data.')

        print(self.length)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.items[idx]
        uniid = data['uniid']
        pre_text = data['text']
        img = data['image']
        image = None
        if img and (not self.skip_img):
            for base64_string in img:
                if base64_string is not None:
                    image = Image.open(base64_string).convert('RGB').resize((224,224))
        pre_img = [None, None, None, None]
        if image:
            pre_img[data['random_idx']] = image
        if self.skip_txt:
            pre_text = ' '
        image = Image.fromarray(self.tokenize(pre_text, pre_img))
        images = self.transforms(image)
        return images, uniid


def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end


def get_all_embed(model, dataloader, device, amp=True):
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress
    for batch_images, t_id, inds in tqdm(dataloader):
        batch_images = batch_images.to(device)
        with torch.no_grad(), autocast():
            batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
        batch_images_emb_list.append(batch_images_emb.cpu())
        texts_image_index.extend(t_id)

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_image_index = torch.stack(texts_image_index)
    return images_emb, texts_image_index.cpu().tolist()


def evaluate(query_emb, support_emb, prefix_mask, positive_pairs, device, recall_k_list=[1]):
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

    scores = query_emb @ support_emb.t()
    if prefix_mask is not None:
        scores += prefix_mask


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
        metrics[f"retrieval_recall@{recall_k}"] = (batchify(recall_at_k_sample, scores, positive_pairs, 1024, device, k=recall_k)>0).float()

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


def recall_at_k_sample(scores, positive_pairs, k):
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
    res = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # print(res.shape)
    return res


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


from open_clip import create_model_and_transforms


def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None):
    model, _, transform = create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)
    model = model.cuda()
    return model, transform


def get_pass_rate(QIN_emb, IN_emb, qin_item_ids, in_item_ids):
    scores = QIN_emb @ IN_emb.t()
    prefix_mask = torch.zeros_like(scores)
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    # for candidactes pool
    doc_ids = [item // 10000 for item in in_item_ids]
    text_ids = [((item % 10000) // 10) for item in in_item_ids]
    
    for query_idx, q_id in enumerate(qin_item_ids):
        qdid = q_id // 10000
        qtid = ((q_id % 10000) // 10)
        prefix = []
        match_mask = []
        # find query's gt
        for idx, iid in enumerate(doc_ids):
            if (doc_ids[idx] == qdid):
                if (text_ids[idx] <= qtid):
                    prefix.append(idx) 
                if (text_ids[idx] == qtid+1):
                    match_mask.append(idx)
        # mask snippts before query 
        prefix_mask[query_idx, prefix] = -100
        positive_pairs[query_idx, match_mask] = True
    
    in_2_in = evaluate(QIN_emb, IN_emb, prefix_mask, positive_pairs, "cuda")
    qin_res = in_2_in['retrieval_recall@1']
    group_score = {}
    # group socre for each doc
    for query_idx, q_id in enumerate(qin_item_ids):
        qdid = q_id // 10000
        qtid = ((q_id % 10000) // 10)
        if group_score.get(qdid, None) is None:
            group_score[qdid] = [0, 0, 0, 0]
        group_score[qdid][qtid] = qin_res[query_idx]
    pass_rate = [0,0,0,0]
    # get n round success rate
    for key in group_score.keys():
        for i in range(1, 4):
            group_score[key][i] *= group_score[key][i-1]
        for i in range(4):
            pass_rate[i] += group_score[key][i]
    return [p / len(group_score.keys()) for p in pass_rate]



valid_data = get_val_data()
dataloader_init = None

for ckpt in ckpt_list:
    print(ckpt)
    model, transform = load_open_clip("ViT-B-16-448", ckpt)
    if not dataloader_init:
        IN_dataloader = torch.utils.data.DataLoader(
                        SeqDocDataset(valid_data, transform), batch_size=128, 
                        shuffle=False, num_workers=8, 
                    )
        Txt_dataloader = torch.utils.data.DataLoader(
                        SeqDocDataset(valid_data, transform, skip_img=True), batch_size=128, 
                        shuffle=False, num_workers=8, 
                    )
        QueryIN_dataloader = torch.utils.data.DataLoader(
                        SeqDocDataset(valid_data, transform, fliter_len=5), batch_size=128, 
                        shuffle=False, num_workers=8, 
                    )
        QueryTxt_dataloader = torch.utils.data.DataLoader(
                        SeqDocDataset(valid_data, transform, skip_img=True, fliter_len=5), batch_size=128, 
                        shuffle=False, num_workers=8, 
                    )
    res = {}
    if True:
        IN_emb, in_item_ids = get_all_embed(model, IN_dataloader, "cuda")
        print("load IN Feats")

        QIN_emb, qin_item_ids = get_all_embed(model, QueryIN_dataloader, "cuda")
        print("load Query IN Feats")
        
        res["IN@IN"] = get_pass_rate(QIN_emb, IN_emb, qin_item_ids, in_item_ids)

    from tabulate import tabulate
    headers = ["Type", "IR@1", "TR@1", "IR@5", "TR@5"]
    rows = []
    for key in res.keys():
        rows.append([key]+ list(res[key])) 
    # print(rows, headers)
    markdown_table = tabulate(rows, headers, tablefmt="pipe", floatfmt=".6f")
    print()
    print(markdown_table)
    print()
