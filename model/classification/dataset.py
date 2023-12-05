# Standard Library Modules
import os
import sys
import json
import pickle
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
import torch
from torch.utils.data.dataset import Dataset
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import get_huggingface_model_name

class ClassificationDataset(Dataset):
    def __init__(self, args, data_path:str) -> None:
        super(ClassificationDataset, self).__init__()
        self.args = args
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []
        self.num_classes = data_['num_classes']

        model_name = get_huggingface_model_name(args.model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        for idx in tqdm(range(len(data_['input_text'])), desc=f'Loading data from {data_path}'):
            self.data_list.append({
                'input_text': data_['input_text'][idx],
                'label': data_['labels'][idx],
                'soft_label': data_['soft_labels'][idx],
                'index': idx,
            })

        del data_

    def __getitem__(self, idx:int) -> dict:
        # Tokenize input text
        input_tokenized = self.tokenizer(
            self.data_list[idx]['input_text'],
            padding='max_length',
            truncation=True,
            max_length=self.args.max_seq_len,
            return_tensors='pt',
        )

        input_tokenized = {k: v.squeeze(0) for k, v in input_tokenized.items()}

        return {
            'input_data': input_tokenized,
            'label': torch.tensor(self.data_list[idx]['label'], dtype=torch.long),
            'soft_label': torch.tensor(self.data_list[idx]['soft_label'], dtype=torch.float),
            'index': idx,
        }

    def __len__(self) -> int:
        return len(self.data_list)

class ZeroShotDataset(Dataset):
    def __init__(self, args, data_path:str) -> None:
        super(ZeroShotDataset, self).__init__()
        self.args = args
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        # Load prompt json file
        with open(args.cls_prompt_path, 'r') as f:
            self.prompt_dict = json.load(f)
        self.placeholders = self.prompt_dict["placeholders"]
        self.instruction = self.prompt_dict["prompts"]["instruction"]
        self.label_list = [self.prompt_dict["labels"][str(i)] for i in range(len(self.prompt_dict["labels"]))]

        self.data_list = []
        self.num_classes = data_['num_classes']

        self.model_name = get_huggingface_model_name(args.model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        for idx in tqdm(range(len(data_['input_text'])), desc=f'Loading data from {data_path}'):
            self.data_list.append({
                'input_text': data_['input_text'][idx],
                'label': data_['labels'][idx],
                'index': idx,
            })

        del data_

    def __getitem__(self, idx:int) -> dict:
        input_prompt = self._build_prompt(self.data_list[idx]['input_text'])
        input_tokenized_list = []
        for each_prompt in input_prompt:
            input_tokenized = self.tokenizer(
                each_prompt,
                padding='max_length',
                truncation=True,
                max_length=self.args.max_seq_len,
                return_tensors='pt',
            )
            input_tokenized = {k: v.squeeze(0) for k, v in input_tokenized.items()}
            input_tokenized_list.append(input_tokenized)

        return {
            'input_list': input_tokenized_list,
            'label': torch.tensor(self.data_list[idx]['label'], dtype=torch.long),
            'index': idx,
        }

    def __len__(self) -> int:
        return len(self.data_list)

    def _build_prompt(self, input_text: str) -> list:
        input_prompt = []
        for each_label in self.label_list:
            prompt = self.instruction.replace(self.placeholders["label"], each_label)
            prompt = prompt.replace(self.placeholders["text"], input_text)
            input_prompt.append(prompt)

        return input_prompt

class GenerationDataset(Dataset):
    def __init__(self, args, label:int) -> None:
        super(GenerationDataset, self).__init__()
        self.args = args
        # Load prompt json file
        with open(args.gen_prompt_path, 'r') as f:
            self.prompt_dict = json.load(f)
        self.placeholders = self.prompt_dict["placeholders"]
        self.instructions = self.prompt_dict["prompts"]
        self.label_list = [self.prompt_dict["labels"][str(i)] for i in range(len(self.prompt_dict["labels"]))]

        self.model_name = get_huggingface_model_name(args.model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # There is no data -> fill with instructions with desired amount of data
        self.data_list = []
        for i in range(self.args.gen_amount // len(self.label_list)):
            self.data_list.append({
                'label': self.label_list[label],
                'index': i,
            })

    def __getitem__(self, idx:int) -> dict:
        return {
            'label': self.data_list[idx]['label'],
            'label_idx': self.label_list.index(self.data_list[idx]['label']),
            'index': idx,
        }

    def __len__(self) -> int:
        return len(self.data_list)

    def build_prompt(self, stage:int, input_text:str=None, label:str=None, topic:str=None) -> str:
        if stage == 1:
            prompt = self.instructions[f"stage{stage}"].replace(self.placeholders["label"], label)
        elif stage == 3:
            prompt = self.instructions[f"stage{stage}"]
        elif stage == 4:
            prompt = self.instructions[f"stage{stage}"].replace(self.placeholders["label"], input_text)
            prompt = prompt.replace(self.placeholders["topic"], topic)
        elif stage in [2, 5]:
            prompt = self.instructions[f"stage{stage}"].replace(self.placeholders["label"], label)
            prompt = prompt.replace(self.placeholders["text"], input_text)

        return prompt
