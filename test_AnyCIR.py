import ast
import json
import logging
import math
import os
import torch
from contextlib import suppress
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from training.render_img_text_np import get_pp_render_text
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import random
from pathlib import Path
import orjson
import warnings
warnings.filterwarnings("ignore")

from open_clip import create_model_and_transforms
OBLICES_PATH = "path_to_oblices_ann"
OBLICES_SEEK_MAP_PATH = "path_to_oblices_ann_seek_map"
ckpt_list = ["path_to_ckpt"]

def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None):
    model, _, transform = create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)
    model = model.cuda()
    return model, transform

def get_valid_data():
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
        pre_flag, nxt_flag = 0, 0
        for url in data['pre_img']:
            if url is not None:
                pre_flag = 1
                continue
        for url in data['next_img']:
            if url is not None:
                nxt_flag = 1
                continue
        if pre_flag == 0 or nxt_flag == 0:
            continue
        data['pre_img_idx'] = random.sample([0, 1, 2, 3], 1)[0]
        data['next_img_idx'] = random.sample([0, 1, 2, 3], 1)[0]
        valid_data.append(data)
    return valid_data

class PairDocDataset(Dataset):

    def __init__(self, valid_data, transforms, 
        skip_pre_img=False, skip_next_img=False,
        skip_pre_txt=False, skip_next_txt=False
        ):
        self.valid_data = valid_data

        self.length = len(self.valid_data)
        self.transforms = transforms
        logging.debug('Done loading data.')
        self.skip_pre_img = skip_pre_img
        self.skip_next_img = skip_next_img
        self.skip_pre_txt = skip_pre_txt
        self.skip_next_txt = skip_next_txt
        self.tokenize = get_pp_render_text(max_chars=1100)

    def __len__(self):
        # return 100
        return self.length

    def __getitem__(self, idx):
        data = self.valid_data[idx]

        pre_text = data['pre_text']
        pre_img = data['pre_img']
        next_text = data['next_text']
        next_img = data['next_img']
        image = None
        base_url = ""
        if pre_img and (not self.skip_pre_img):
            for base64_string in pre_img:
                if base64_string is not None:
                    image = Image.open(base64_string).convert('RGB').resize((224,224))
                    break
        pre_img = [None, None, None, None]
        if image:
            pre_img[data['pre_img_idx']] = image
        if self.skip_pre_txt:
            pre_text = ' '
        image = Image.fromarray(self.tokenize(pre_text, pre_img))
        images = self.transforms(image)
        image = None
        if next_img and (not self.skip_next_img):
            for base64_string in next_img:
                if base64_string is not None:
                    image = Image.open(base64_string).convert('RGB').resize((224,224))
                    break
        next_img = [None, None, None, None]
        if image:
            next_img[data['next_img_idx']] = image
        if self.skip_next_txt:
            next_text = ' '
        texts = Image.fromarray(self.tokenize(next_text, next_img))
        texts = self.transforms(texts)
        return images, texts


def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end

def evaluate(model, dataloader, device, amp=True, recall_k_list=[1,5]):
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
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress
    for batch_images, batch_texts, inds in tqdm(dataloader):
        batch_images = batch_images.to(device)
        batch_texts = batch_texts.to(device)
        # # tokenize all texts in the batch
        # batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        # # store the index of image for each text
        batch_texts_image_index = [ind for ind in inds]
        # print(len(inds))
        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
            batch_texts_emb = F.normalize(model.encode_text(batch_texts), dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)
        
    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)
    # print(images_emb.size())
    # print(texts_emb.size())
    # get the score for each text and image pair
    scores  = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
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
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, 1024, device, k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, 1024, device, k=recall_k)>0).float().mean().item()

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
        # print("x size: ", x.size())
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


data = get_valid_data()

for ckpt in ckpt_list:
    print(ckpt)
    print()
    model, transform = load_open_clip("ViT-B-16-448", ckpt)
    res = {}

    dataset = PairDocDataset(data, transform)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=128, 
                    shuffle=False, num_workers=16, 
                )
    print(f"test file: {num_k}k", len(dataset))
    res["IN-IN"] = evaluate(model, dataloader, "cuda")
    # print("IN-IN", res["IN-IN"])

    dataset = PairDocDataset(data, transform,
            skip_pre_img=False, skip_next_img=True,
            skip_pre_txt=False, skip_next_txt=False)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=128, 
                    shuffle=False, num_workers=16, 
                )
    res["IN-Txt"] = evaluate(model, dataloader, "cuda")
    # print("IN-Txt", res["IN-Txt"])

    dataset = PairDocDataset(data, transform,
            skip_pre_img=False, skip_next_img=False,
            skip_pre_txt=False, skip_next_txt=True)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=128, 
                    shuffle=False, num_workers=16, 
                )
    res["IN-Img"] = evaluate(model, dataloader, "cuda")
    # print("IN-Img", res["IN-Img"])

    dataset = PairDocDataset(data, transform,
            skip_pre_img=True, skip_next_img=False,
            skip_pre_txt=False, skip_next_txt=False)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=128, 
                    shuffle=False, num_workers=16, 
                )
    res["Txt-IN"] = evaluate(model, dataloader, "cuda")
    # print("Txt-IN", res["Txt-IN"])

    dataset = PairDocDataset(data, transform,
            skip_pre_img=True, skip_next_img=True,
            skip_pre_txt=False, skip_next_txt=False)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=128, 
                    shuffle=False, num_workers=16, 
                )
    res["Txt-Txt"] = evaluate(model, dataloader, "cuda")
    # print("Txt-Txt", res["Txt-Txt"])

    dataset = PairDocDataset(data, transform,
            skip_pre_img=True, skip_next_img=False,
            skip_pre_txt=False, skip_next_txt=True)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=128, 
                    shuffle=False, num_workers=16, 
                )
    res["Txt-Im"] = evaluate(model, dataloader, "cuda")
    # print("Txt-Im", res["Txt-Im"])

    dataset = PairDocDataset(data, transform,
            skip_pre_img=False, skip_next_img=False,
            skip_pre_txt=True, skip_next_txt=False)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=128, 
                    shuffle=False, num_workers=16, 
                )
    res["Img-IN"] = evaluate(model, dataloader, "cuda")
    # print("Img-IN", res["Img-IN"])



    dataset = PairDocDataset(data, transform,
            skip_pre_img=False, skip_next_img=True,
            skip_pre_txt=True, skip_next_txt=False)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=128, 
                    shuffle=False, num_workers=16, 
                )
    res["Im-Txt"] = evaluate(model, dataloader, "cuda")
    # print("Im-Txt", res["Im-Txt"])

    dataset = PairDocDataset(data, transform,
            skip_pre_img=False, skip_next_img=False,
            skip_pre_txt=True, skip_next_txt=True)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=128, 
                    shuffle=False, num_workers=16, 
                )
    res["Im-Im"] = evaluate(model, dataloader, "cuda")
    # print("Im-Im", res["Im-Im"])

    from tabulate import tabulate
    headers = ["Type", "PreNxt@1", "NxtPre@1", "PreNxt@5", "NxtPre@5"]
    rows = []
    for key in res.keys():
        rows.append([key]+ list(res[key].values())) 
    markdown_table = tabulate(rows, headers, tablefmt="pipe", floatfmt=".6f")
    print()
    print(markdown_table)
    print()