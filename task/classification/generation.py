# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # This prevents tokenizers from taking all cpus
import sys
import random
import pickle
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import pandas as pd
# Pytorch Modules
import torch
torch.set_num_threads(2) # This prevents Pytorch from taking all cpus
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# HuggingFace Modules
from transformers import AutoModelForCausalLM
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.classification.dataset import GenerationDataset
from utils.utils import TqdmLoggingHandler, write_log, check_path, get_torch_device, get_huggingface_model_name

def generation(args: argparse.Namespace) -> None:
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

    # Load dataset and define dataloader
    dataset_gen, dataloader_gen = {}, {}
    dataset_gen['negative'] = GenerationDataset(args, label=0)
    dataset_gen['positive'] = GenerationDataset(args, label=1)
    dataloader_gen['negative'] = DataLoader(dataset_gen['negative'], batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True, drop_last=False)
    dataloader_gen['positive'] = DataLoader(dataset_gen['positive'], batch_size=args.batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True, drop_last=False)
    label_list = dataset_gen['positive'].label_list
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    write_log(logger, "Loaded data successfully")

    # Get model instance
    write_log(logger, "Building model")
    if args.model_type in ['gpt2', 'gpt2_large', 'gpt2_xl', 'opt', 'bloom']:
        model_name = get_huggingface_model_name(args.model_type)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = dataset_gen['negative'].tokenizer
    else:
        raise NotImplementedError(f"Model type {args.model_type} is not implemented.")

    # Start generation
    model = model.eval()
    generated_data = []
    while len(generated_data) < args.gen_amount // len(label_list):
        # Generate negative data
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_gen['negative'], total=len(dataloader_gen['negative']), desc="Generation-NEG", position=0, leave=True)):
            # Gen - Get input data
            labels = data_dicts['label']
            label_idx = data_dicts['label_idx']

            # Gen - STAGE 1
            input_prompt = []
            for label in labels:
                input_prompt.append(dataset_gen['negative'].build_prompt(stage=1, label=label))
            input_tokenized = tokenizer(input_prompt, return_tensors='pt', padding=True).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_tokenized['input_ids'],
                    attention_mask=input_tokenized['attention_mask'],
                    max_length=args.max_seq_len,
                    top_k=args.gen_top_k,
                    top_p=args.gen_top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=args.gen_temperature,
                    early_stopping=True,
                )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs_stage1 = [process_output(output) for output in outputs]

            # Gen - STAGE 2 - First Pseudo-labeling to filter out sentences that are not related to the label
            if args.gen_relabel != 'none':
                input_prompt = []
                for each_label in label_list:
                    for each_output in outputs_stage1:
                        input_prompt.append(dataset_gen['negative'].build_prompt(stage=2, input_text=each_output, label=each_label))
                input_tokenized = tokenizer(input_prompt, return_tensors='pt', padding=True).to(device)

                # Divide into original batch_size batches, to prevent OOM
                input_tokenized_list = []
                for i in range(input_tokenized['input_ids'].size(0) // args.batch_size):
                    input_tokenized_list.append({k: v[i * args.batch_size: (i + 1) * args.batch_size] for k, v in input_tokenized.items()})

                output_logits_list = []
                for input_tokenized_subbatch in input_tokenized_list:
                    with torch.no_grad():
                        output_logits_list.append(model(**input_tokenized_subbatch))
                shifted_logits_list = [output.logits[:, :-1, :].contiguous() for output in output_logits_list]
                shifted_labels_list = [input_tokenized_subbatch['input_ids'][:, 1:].contiguous() for input_tokenized_subbatch in input_tokenized_list]

                loss_list = [loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                                 shifted_labels.view(-1)).view(shifted_labels.size())
                         for shifted_logits, shifted_labels in zip(shifted_logits_list, shifted_labels_list)]
                avg_loss_list = [loss.sum(-1) / (loss > 0).sum(-1).float() for loss in loss_list]
                if len(avg_loss_list) == 0:
                    continue

                pred = torch.stack(avg_loss_list, dim=1)
                prob = torch.softmax(pred / args.gen_relabel_temperature, dim=-1)
                # Small loss -> More appropriate sentence -> More likely to be related to the label
                # We need to reverse prob, so that the sentence with the highest probability is the one with the smallest loss
                prob = 1 - prob

                # We will only keep the sentences that have high probability that exceeds the threshold
                # This is to prevent the model from generating sentences that are not related to the label
                prob_max = prob.max(dim=-1)[0]
                prob_max_exceed = prob_max > (1 / len(label_list)) + args.gen_relabel_threshold
                prob_max_match = prob.argmax(dim=-1) == torch.tensor(0).to(device) # 0: Negative

                # Only save sentences that exceed the threshold
                outputs_stage2 = []
                for i in range(len(prob_max_exceed)):
                    if prob_max_exceed[i] and prob_max_match[i]:
                        outputs_stage2.append({
                            'sentence': outputs_stage1[i],
                            'label_soft': [round(prob[i][j].item(), 2) for j in range(len(prob[i]))],
                            'label_hard': prob[i].argmax().item(),
                            'label_noisy': label_idx[i].item(),
                        })
            else:
                outputs_stage2 = []
                for i in range(len(outputs_stage1)):
                    outputs_stage2.append({
                        'sentence': outputs_stage1[i],
                        'label_soft': [0.0 for j in range(len(label_list))],
                        'label_hard': -1,
                        'label_noisy': label_idx[i].item(),
                    })
            if len(outputs_stage2) == 0:
                continue # If there is no sentence that exceeds the threshold, skip the rest of the process
            else:
                # tqdm.write(str(outputs_stage2))
                generated_data.extend(outputs_stage2)
    if len(generated_data) > args.gen_amount // len(label_list):
        generated_data = generated_data[:args.gen_amount // len(label_list)]

    while len(generated_data) < args.gen_amount:
        # Generate positive data
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_gen['positive'], total=len(dataloader_gen['positive']), desc="Generation-POS", position=0, leave=True)):
            # Gen - Get input data
            labels = data_dicts['label']
            label_idx = data_dicts['label_idx']

            # Gen - STAGE 1
            input_prompt = []
            for label in labels:
                input_prompt.append(dataset_gen['positive'].build_prompt(stage=1, label=label))
            input_tokenized = tokenizer(input_prompt, return_tensors='pt', padding=True).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_tokenized['input_ids'],
                    attention_mask=input_tokenized['attention_mask'],
                    max_length=args.max_seq_len,
                    top_k=args.gen_top_k,
                    top_p=args.gen_top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=args.gen_temperature,
                    early_stopping=True,
                )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs_stage1 = [process_output(output) for output in outputs]

            # Gen - STAGE 2 - Pseudo-labeling to filter out sentences that are not related to the label
            if args.gen_relabel != 'none':
                input_prompt = []
                for each_label in label_list:
                    for each_output in outputs_stage1:
                        input_prompt.append(dataset_gen['positive'].build_prompt(stage=2, input_text=each_output, label=each_label))
                input_tokenized = tokenizer(input_prompt, return_tensors='pt', padding=True).to(device)

                # Divide into original batch_size batches, to prevent OOM
                input_tokenized_list = []
                for i in range(input_tokenized['input_ids'].size(0) // args.batch_size):
                    input_tokenized_list.append({k: v[i * args.batch_size: (i + 1) * args.batch_size] for k, v in input_tokenized.items()})

                output_logits_list = []
                for input_tokenized_subbatch in input_tokenized_list:
                    with torch.no_grad():
                        output_logits_list.append(model(**input_tokenized_subbatch))
                shifted_logits_list = [output.logits[:, :-1, :].contiguous() for output in output_logits_list]
                shifted_labels_list = [input_tokenized_subbatch['input_ids'][:, 1:].contiguous() for input_tokenized_subbatch in input_tokenized_list]

                loss_list = [loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                                 shifted_labels.view(-1)).view(shifted_labels.size())
                         for shifted_logits, shifted_labels in zip(shifted_logits_list, shifted_labels_list)]
                avg_loss_list = [loss.sum(-1) / (loss > 0).sum(-1).float() for loss in loss_list]
                if len(avg_loss_list) == 0:
                    continue

                pred = torch.stack(avg_loss_list, dim=1)
                prob = torch.softmax(pred / args.gen_relabel_temperature, dim=-1)
                # Small loss -> More appropriate sentence -> More likely to be related to the label
                # We need to reverse prob, so that the sentence with the highest probability is the one with the smallest loss
                prob = 1 - prob

                # We will only keep the sentences that have high probability that exceeds the threshold
                # This is to prevent the model from generating sentences that are not related to the label
                prob_max = prob.max(dim=-1)[0]
                prob_max_exceed = prob_max > (1 / len(label_list)) + args.gen_relabel_threshold
                prob_max_match = prob.argmax(dim=-1) == torch.tensor(1).to(device) # 1: Positive

                # Only save sentences that exceed the threshold
                outputs_stage2 = []
                for i in range(len(prob_max_exceed)):
                    if prob_max_exceed[i] and prob_max_match[i] and outputs_stage1[i] != "The text in positive sentiment is":
                        # Some heuristic to prevent the model from generating the same sentence
                        outputs_stage2.append({
                            'sentence': outputs_stage1[i],
                            'label_soft': [round(prob[i][j].item(), 2) for j in range(len(prob[i]))],
                            'label_hard': prob[i].argmax().item(),
                            'label_noisy': label_idx[i].item(),
                        })
            else:
                outputs_stage2 = []
                for i in range(len(outputs_stage1)):
                    outputs_stage2.append({
                        'sentence': outputs_stage1[i],
                        'label_soft': [0.0 for j in range(len(label_list))],
                        'label_hard': -1,
                        'label_noisy': label_idx[i].item(),
                    })
            if len(outputs_stage2) == 0:
                continue # If there is no sentence that exceeds the threshold, skip the rest of the process
            else:
                # tqdm.write(str(outputs_stage2))
                generated_data.extend(outputs_stage2)
    if len(generated_data) > args.gen_amount:
        generated_data = generated_data[:args.gen_amount]

    # Transform generated data into data_dict format
    data_dict = {
        'train_NL': { # Noisy Label
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': len(label_list),
        },
        'valid_NL': {
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': len(label_list),
        },
        'train_SL': { # Soft Label
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': len(label_list),
        },
        'valid_SL': {
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': len(label_list),
        },
        'train_HL': { # Hard Label
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': len(label_list),
        },
        'valid_HL': {
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': len(label_list),
        },
    }

    # Shuffle generated data
    random.shuffle(generated_data)

    # Split generated data into train and valid
    valid_amount = int(len(generated_data) * args.train_valid_split) # 0.1
    train_data = generated_data[valid_amount:]
    valid_data = generated_data[:valid_amount]

    # Save each data into data_dict
    for i in range(len(train_data)):
        data_dict['train_NL']['input_text'].append(train_data[i]['sentence'])
        data_dict['train_HL']['input_text'].append(train_data[i]['sentence'])
        data_dict['train_SL']['input_text'].append(train_data[i]['sentence'])

        data_dict['train_NL']['labels'].append(train_data[i]['label_noisy'])
        data_dict['train_HL']['labels'].append(train_data[i]['label_hard'])
        data_dict['train_SL']['labels'].append(train_data[i]['label_hard'])

        data_dict['train_SL']['soft_labels'].append(train_data[i]['label_soft'])
        soft_label_for_noisy = [0.0] * len(label_list)
        soft_label_for_noisy[train_data[i]['label_noisy']] = 1.0
        data_dict['train_NL']['soft_labels'].append(soft_label_for_noisy)
        soft_label_for_hard = [0.0] * len(label_list)
        soft_label_for_hard[train_data[i]['label_hard']] = 1.0
        data_dict['train_HL']['soft_labels'].append(soft_label_for_hard)

    for i in range(len(valid_data)):
        data_dict['valid_NL']['input_text'].append(valid_data[i]['sentence'])
        data_dict['valid_HL']['input_text'].append(valid_data[i]['sentence'])
        data_dict['valid_SL']['input_text'].append(valid_data[i]['sentence'])

        data_dict['valid_NL']['labels'].append(valid_data[i]['label_noisy'])
        data_dict['valid_HL']['labels'].append(valid_data[i]['label_hard'])
        data_dict['valid_SL']['labels'].append(valid_data[i]['label_hard'])

        data_dict['valid_SL']['soft_labels'].append(valid_data[i]['label_soft'])
        soft_label_for_noisy = [0.0] * len(label_list)
        soft_label_for_noisy[valid_data[i]['label_noisy']] = 1.0
        data_dict['valid_NL']['soft_labels'].append(soft_label_for_noisy)
        soft_label_for_hard = [0.0] * len(label_list)
        soft_label_for_hard[valid_data[i]['label_hard']] = 1.0
        data_dict['valid_HL']['soft_labels'].append(soft_label_for_hard)

    # Save data_dict as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset)
    check_path(preprocessed_path)
    if 'zerogen' in args.generation_type:
        for split in ['train', 'valid']:
            with open(os.path.join(preprocessed_path, f'{split}_ZG_NL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'), 'wb') as f:
                pickle.dump(data_dict[f'{split}_NL'], f)
            print(f"saved {split}_ZG_NL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl")
            if args.gen_relabel != 'none':
                with open(os.path.join(preprocessed_path, f'{split}_ZG_SL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'), 'wb') as f:
                    pickle.dump(data_dict[f'{split}_SL'], f)
                print(f"saved {split}_ZG_SL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl")
                with open(os.path.join(preprocessed_path, f'{split}_ZG_HL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'), 'wb') as f:
                    pickle.dump(data_dict[f'{split}_HL'], f)
                print(f"saved {split}_ZG_HL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl")
    elif 'unigen' in args.generation_type:
        preprocessed_path = os.path.join(args.preprocess_path, args.task) # Unigen is not dataset-specific
        for split in ['train', 'valid']:
            with open(os.path.join(preprocessed_path, f'{split}_UG_NL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'), 'wb') as f:
                pickle.dump(data_dict[f'{split}_NL'], f)
            print(f"saved {split}_UG_NL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl")
            if args.gen_relabel != 'none':
                with open(os.path.join(preprocessed_path, f'{split}_UG_SL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'), 'wb') as f:
                    pickle.dump(data_dict[f'{split}_SL'], f)
                print(f"saved {split}_UG_SL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl")
                with open(os.path.join(preprocessed_path, f'{split}_UG_HL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'), 'wb') as f:
                    pickle.dump(data_dict[f'{split}_HL'], f)
                print(f"saved {split}_UG_HL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl")


def process_output(output_text: str) -> str:
    # remove prompt first
    output_text = output_text.split(": \"")[1]

    # if the sentence has \n, delete it and words after it
    if "\n" in output_text:
        output_text = output_text.split("\n")[0]
    if "\"" in output_text:
        output_text = output_text.split("\"")[0]

    return output_text
