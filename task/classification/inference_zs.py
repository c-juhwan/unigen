# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # This prevents tokenizers from taking all cpus
import sys
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
from transformers import GPT2LMHeadModel
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.classification.dataset import ZeroShotDataset
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, get_huggingface_model_name

def inference(args: argparse.Namespace) -> None:
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

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Load dataset and define dataloader
    write_log(logger, "Loading dataset")
    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['test'] = ZeroShotDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset, f'test_ORI.pkl'))

    dataloader_dict['test'] = DataLoader(dataset_dict['test'], batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=False, pin_memory=True, drop_last=False)

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_dict['test'])} / {len(dataloader_dict['test'])}")

    # Get model instance
    write_log(logger, "Building model")
    if args.model_type in ['gpt2', 'gpt2_large', 'gpt2_xl']:
        model_name = get_huggingface_model_name(args.model_type)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = dataset_dict['test'].tokenizer
    else:
        raise NotImplementedError(f"Model type {args.model_type} is not implemented.")

    # Load Wandb
    if args.use_wandb:
        import wandb
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                   name=get_wandb_exp_name(args) + f' - Test',
                   config=args,
                   notes=args.description,
                   tags=["TEST",
                         f"Dataset: {args.task_dataset}",
                         f"Model: {args.model_type}",
                         f"Training: {args.training_type}"])

    # Test - Start testing on test set
    loss_fn = nn.CrossEntropyLoss(reduce=False, ignore_index=tokenizer.pad_token_id)
    model = model.eval()
    test_acc_cls = 0
    test_f1_cls = 0
    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['test'], total=len(dataloader_dict['test']), desc="Testing on TEST Set", position=0, leave=True)):
        # Test - Get input data
        input_list = data_dicts['input_list']
        input_list = [{k: v.to(device) for k, v in input_dict.items()} for input_dict in input_list]
        labels = data_dicts['label'].to(device)

        outputs_list = []
        with torch.no_grad():
            outputs_list = [model(**input_dict) for input_dict in input_list]
        shifted_logit_list = [outputs.logits[:, :-1, :].contiguous() for outputs in outputs_list]
        shifted_label_list = [input_dict['input_ids'][:, 1:].contiguous() for input_dict in input_list]

        loss_list = [loss_fn(shifted_logit.view(-1, shifted_logit.size(-1)),
                             shifted_label.view(-1)).view(shifted_label.size())
                     for shifted_logit, shifted_label in zip(shifted_logit_list, shifted_label_list)]
        avg_loss_list = [loss.sum(-1) / (loss > 0).sum(-1).float() for loss in loss_list]

        # Select the sentence with lower loss as prediction
        # Assign 1 to positive sentiment and 0 to negative sentiment
        pred = torch.stack(avg_loss_list, dim=1).argmin(dim=1)

        batch_acc = (pred == labels).sum().item() / len(labels)
        batch_f1 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='macro')
        test_acc_cls += batch_acc
        test_f1_cls += batch_f1

    # Test - Check metric
    test_acc_cls /= len(dataloader_dict['test'])
    test_f1_cls /= len(dataloader_dict['test'])

    # Final - End of testing
    write_log(logger, f"Done! - TEST SET - Acc: {test_acc_cls:.4f} - F1: {test_f1_cls:.4f}")
    if args.use_tensorboard:
        writer.add_scalar('TEST/Acc', test_acc_cls, 0)
        writer.add_scalar('TEST/F1', test_f1_cls, 0)
        writer.close()
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'Dataset': [args.task_dataset],
            'Model': [args.model_type],
            # 'Training': [args.training_type],
            # 'Gen_TopK': [args.gen_top_k],
            # 'Gen_TopP': [args.gen_top_p],
            # 'Gen_Temp': [args.gen_temperature],
            # 'SupCon_Weight': [args.supcon_loss_weight],
            # 'SupCon_Temp': [args.supcon_temperature],
            # 'SupCon_MBSize': [args.supcon_memory_bank_size],
            # 'SupCon_MMTau': [args.supcon_momentum_tau],
            'Cls_Prompt': [args.cls_prompt],
            # 'Gen_Prompt': [args.gen_prompt],
            'Test_Acc': [test_acc_cls],
            'Test_F1': [test_f1_cls],
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({'TEST_Result': wandb_table})

        wandb.finish()
