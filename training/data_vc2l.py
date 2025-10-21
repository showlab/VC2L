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

hvd = None
from PIL import Image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import cv2
import random
from pathlib import Path
import orjson
import nlpaug.augmenter.word as naw

IMAGE_PATH = "path_to_mmc4_image"
ANN_PATH = "path_to_mmc4_annotation"
SEEK_MAP_PATH = "path_to_mmc4_annotation_seek_map"

class VC2LDataset(Dataset):

    def __init__(self, 
        input_filename, 
        transforms,
        drop_rate=0,
        aug_text=0,
        long_range=0,
        use_224=False
        ):
        logging.debug(f'Loading data from {input_filename}.')
        self.root_dir = IMAGE_PATH
        self.path_mmc4 = Path(ANN_PATH)
        seek_map_mmc4 = Path(SEEK_MAP_PATH)
        self.seek_map_mmc4 = orjson.loads(seek_map_mmc4.open().read())
        self.length = len(self.seek_map_mmc4)
        self.dataset_file_mmc4 = None
        self.transforms = transforms
        logging.debug('Done loading data.')
        self.long_range = long_range
        self.drop_rate = drop_rate
        self.aug_text = aug_text
        logging.info(f'Drop Rate {self.drop_rate}.')
        logging.info(f'Aug Text Rate {self.aug_text}.')
        logging.info(f'long range Rate {self.long_range}.')
        self.tokenize = get_pp_render_text()
        self.aug = naw.RandomWordAug(action='crop')
        self.return_weight = False
        self.use_224 = use_224
        logging.info(f'Using 224: {self.use_224}.')
    def __len__(self):
        return self.length

    def random_crop(self, text_list):
        list_len = len(text_list)
        while True:
            start_idx, end_idx = sorted(np.random.choice(list_len, 2, replace=False))
            if end_idx - start_idx > 2:
                break
        text_list = text_list[start_idx:end_idx]
        return text_list

    def _aug_text(self, text):
        if (random.random() > self.aug_text) and len(text) > 250:
                text = text.split('.')
                if len(text) > 4:
                    text = self.random_crop(text)
                text = '.'.join(text)
        return text
    
    def __getitem__(self, idx, debug=False):
        position = self.seek_map_mmc4[idx]
        if self.dataset_file_mmc4 is None:
            self.dataset_file_mmc4 = self.path_mmc4.open()
        self.dataset_file_mmc4.seek(position)
        line = self.dataset_file_mmc4.readline()
        data = orjson.loads(line)
        len_text = len(data['text'])
        if (random.random() > self.long_range) or len_text <= 2:
            pre_idx = random.randint(0, len_text-2)
            next_idx = pre_idx + 1
            weight = 1
        else:
            pre_idx = random.randint(0, len_text-3)
            next_idx = pre_idx + 2
            weight = 0.5

        pre_text = data['text'][pre_idx]
        pre_img = data['img'][pre_idx]
        next_text = data['text'][next_idx]
        next_img = data['img'][next_idx]

        random.shuffle(pre_img)
        random.shuffle(next_img)
        if self.aug_text:
            try:
                aug_pre_text = self._aug_text(pre_text)
                if len(aug_pre_text) > 100:
                    pre_text = aug_pre_text
            except:
                pass
            try:
                aug_next_text = self._aug_text(next_text)
                if len(aug_next_text) > 100:
                    next_text = aug_next_text
            except:
                pass
        image = None
        if pre_img:
            for info in pre_img:
                image = None
                try:
                    image_name = info["image_name"]
                    img_url = f'{self.root_dir}/{image_name}'
                    if not os.path.exists(img_url):
                        continue
                    image = Image.open(img_url).convert("RGB").resize((224,224))
                    break
                except Exception as e:
                    continue
        pre_img = [None, None, None, None]
        if image:
            rand_idx = random.sample([0, 1, 2, 3], 1)[0]
            pre_img[rand_idx] = image
            if self.drop_rate and random.random() > self.drop_rate:
                if random.random() > 0.5:
                    pre_text = ' '
                else:
                    pre_img = [None, None, None, None]

        image = Image.fromarray(self.tokenize(pre_text, pre_img))
        if self.use_224:
            image = image.resize((224, 224))
        images = self.transforms(image)
        image = None
        if next_img:
            for info in next_img:
                image = None
                try:
                    image_name = info["image_name"]
                    img_url = f'{self.root_dir}/{image_name}'
                    if not os.path.exists(img_url):
                        continue
                    image = Image.open(img_url).convert("RGB").resize((224,224))
                    # print(img_url)
                    break
                except Exception as e:
                    # print(e)
                    continue
        next_img = [None, None, None, None]
        if image:
            rand_idx = random.sample([0, 1, 2, 3], 1)[0]
            next_img[rand_idx] = image
            if self.drop_rate and random.random() > self.drop_rate:
                if random.random() > 0.5:
                    next_text = ' '
                else:
                    next_img = [None, None, None, None]

        texts = Image.fromarray(self.tokenize(next_text, next_img))
        if self.use_224:
            texts = texts.resize((224, 224))
        texts = self.transforms(texts)
        if self.return_weight:
            return images, texts, weight
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def get_vc2l_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.csv_root 
    # assert input_filename
    dataset = VC2LDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        use_mmc4_text=args.use_mmc4_text,
        drop_rate=args.drop_rate,
        aug_text=args.aug_text,
        long_range=args.long_range,
        use_224=args.use_224,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)
 
def get_dataset_fn(dataset_type):
    if dataset_type == "vc2ldataset":
        return get_vc2l_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    return data
