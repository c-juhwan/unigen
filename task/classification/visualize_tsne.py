# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # This prevents tokenizers from taking all cpus
import sys
import copy
import random
import logging
import argparse
# 3rd-party Modules
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
# Pytorch Modules
import torch
torch.set_num_threads(2) # This prevents Pytorch from taking all cpus
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Huggingface Modules
from transformers import AutoConfig
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.classification.model import ClassificationModel
from model.classification.dataset import ClassificationDataset
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, get_huggingface_model_name

def visualize(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)

    # Define logger and tensorboard writer
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Load various datasets
    dataset_dict, dataloader_dict = {}, {}
    for dataset in ['multi_domain_book', 'multi_domain_dvd', 'multi_domain_kitchen', 'multi_domain_electronics']:
        dataset_dict[dataset] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, dataset, f'test_ORI.pkl'))
    # Load Unigen data
    dataset_dict['unigen'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, # UniGen is not dataset-specific
                                                  f'train_UG_SL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))
    args.num_classes = dataset_dict['unigen'].num_classes

    # Randomly select 2000 samples from unigen
    dataset_dict['unigen'].data_list = random.sample(dataset_dict['unigen'].data_list, 2000)

    # Define dataloader
    dataloader_dict = {}
    for dataset in ['multi_domain_book', 'multi_domain_dvd', 'multi_domain_kitchen', 'multi_domain_electronics', 'unigen']:
        dataloader_dict[dataset] = DataLoader(dataset_dict[dataset], batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=False, pin_memory=True, drop_last=False)

    # Get model instance
    write_log(logger, "Building model")
    model = ClassificationModel(args)

    # Load model weights
    write_log(logger, "Loading model weights")
    final_model_save_name = f'final_model_{args.training_type}_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pt'

    if args.training_type in ['unigen', 'unigen_ablation_noisy_label', 'unigen_ablation_hard_label']:
        load_model_name = os.path.join(args.model_path, args.task, args.model_type, final_model_save_name)
    else:
        load_model_name = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type, final_model_save_name)
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_name}")
    del checkpoint

    # Visualize - t-SNE
    model = model.eval()
    projected_cls_dict, labels_dict = {}, {}

    for dataset in ['multi_domain_book', 'multi_domain_dvd', 'multi_domain_kitchen', 'multi_domain_electronics', 'unigen']:
        write_log(logger, f"Visualizing {dataset} dataset")
        projected_cls_dict[dataset] = []
        labels_dict[dataset] = []
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict[dataset], total=len(dataloader_dict[dataset]), desc=f"Visualizing {dataset} dataset", position=0, leave=True)):
            # Visualize - Get input data
            input_data = {k: v.to(device) for k, v in data_dicts['input_data'].items()}
            labels = data_dicts['label'].to(device)

            # Visualize - Forward pass
            with torch.no_grad():
                _, projected_cls = model(input_ids=input_data['input_ids'],
                                         attention_mask=input_data['attention_mask'],
                                         token_type_ids=input_data['token_type_ids'] if args.model_type not in ['distilbert', 'roberta'] else None)

            projected_cls_dict[dataset].append(projected_cls)
            labels_dict[dataset].append(labels)

    TSNE_model = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=args.seed)

    # Gather all data into one list for training TSNE
    all_projected_cls = []
    for dataset in ['multi_domain_book', 'multi_domain_dvd', 'multi_domain_kitchen', 'multi_domain_electronics', 'unigen']:
        projected_cls_dict[dataset] = torch.cat(projected_cls_dict[dataset], dim=0).cpu().numpy()
        labels_dict[dataset] = torch.cat(labels_dict[dataset], dim=0).cpu().numpy()

        all_projected_cls.append(projected_cls_dict[dataset])
    all_projected_cls = np.concatenate(all_projected_cls, axis=0)

    # Train TSNE
    all_projected_cls_tsne = TSNE_model.fit_transform(all_projected_cls)
    projected_tsne_dict = {}
    for dataset in ['multi_domain_book', 'multi_domain_dvd', 'multi_domain_kitchen', 'multi_domain_electronics', 'unigen']:
        projected_tsne_dict[dataset] = all_projected_cls_tsne[:len(projected_cls_dict[dataset])]
        all_projected_cls_tsne = all_projected_cls_tsne[len(projected_cls_dict[dataset]):]

    # Plot TSNE
    # 1. Plot negative data as 'X' and positive data as 'O'
    # 2. Plot data from different domains with different colors
    # 3. Plot data from UniGen with gray color

    # 1. Plot negative data as 'X' and positive data as 'O'
    plt.figure(figsize=(10, 10))
    color = {'multi_domain_book': 'red', 'multi_domain_dvd': 'blue', 'multi_domain_kitchen': 'green', 'multi_domain_electronics': 'orange', 'unigen': 'gray'}

    for dataset, name in zip(['multi_domain_book', 'multi_domain_dvd', 'multi_domain_kitchen', 'multi_domain_electronics', 'unigen'], ['Book', 'DVD', 'Kitchen', 'Electronics', 'UniGen']):
        plt.scatter(projected_tsne_dict[dataset][labels_dict[dataset] == 0, 0], projected_tsne_dict[dataset][labels_dict[dataset] == 0, 1], color=color[dataset], marker='x', label=f'{name}_Neg')
    for dataset, name in zip(['multi_domain_book', 'multi_domain_dvd', 'multi_domain_kitchen', 'multi_domain_electronics', 'unigen'], ['Book', 'DVD', 'Kitchen', 'Electronics', 'UniGen']):
        plt.scatter(projected_tsne_dict[dataset][labels_dict[dataset] == 1, 0], projected_tsne_dict[dataset][labels_dict[dataset] == 1, 1], color=color[dataset], marker='o', label=f'{name}_Pos')

    plt.legend(ncol=2)
    plt.savefig(os.path.join('./', f'TSNE_Result_{args.seed}.png'))
    plt.close()

    # Print some examples for UniGen
    # for i in range(len(dataset_dict['unigen'].data_list)):
    for i in range(50):
        print(f"UniGen Example {i}: {dataset_dict['unigen'].data_list[i]['input_text']} / {dataset_dict['unigen'].data_list[i]['soft_label']}")