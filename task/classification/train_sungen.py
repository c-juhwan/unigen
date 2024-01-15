# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # This prevents tokenizers from taking all cpus
import sys
import copy
import math
import shutil
import random
import logging
import warnings
import argparse
warnings.filterwarnings('ignore')
# 3rd-party Modules
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
# Pytorch Modules
import torch
torch.set_num_threads(2) # This prevents Pytorch from taking all cpus
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
# Huggingface Modules
from transformers import AutoConfig
# Custom Modules
from .loss import SupConLoss
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.classification.model import ClassificationModel
from model.classification.dataset import ClassificationDataset
from model.optimizer.optimizer import get_optimizer
from model.optimizer.scheduler import get_scheduler
from model.optimizer.bilevel_optimizer import MetaSGD, MetaAdam
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path, get_huggingface_model_name

def sungen_solve(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)

    # Define logger
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Load dataset and define dataloader
    write_log(logger, "Loading data")
    dataset_dict, dataloader_dict = {}, {}

    if args.training_type == 'sungen':
        # SunGen shares the same generated data with ZeroGen
        dataset_dict['train'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset,
                                                                 f'train_ZG_NL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))
    elif args.training_type == 'unigen':
        dataset_dict['train'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, # UniGen is not task-specific
                                                                 f'train_UG_SL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))
    elif args.training_type == 'unigen_ablation_noisy_label':
        dataset_dict['train'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, # UniGen is not task-specific
                                                                 f'train_UG_NL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))
    elif args.training_type == 'unigen_ablation_hard_label':
        dataset_dict['train'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, # UniGen is not task-specific
                                                                 f'train_UG_HL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))

    dataloader_dict['train'] = DataLoader(dataset_dict['train'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=True, pin_memory=True, drop_last=True)
    model_name = get_huggingface_model_name(args.model_type)
    config = AutoConfig.from_pretrained(model_name)
    args.vocab_size = config.vocab_size
    args.num_classes = dataset_dict['train'].num_classes

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Train dataset size / iterations: {len(dataset_dict['train'])} / {len(dataloader_dict['train'])}")

    # Get model instance
    write_log(logger, "Building model")
    model = ClassificationModel(args).to(device)

    # Define loss function
    ce_theta_weight = torch.full([len(dataset_dict['train'])], args.sungen_initial_weight,
                                 dtype=torch.float, requires_grad=True, device=device) # Initialize weight of each training sample for cross entropy loss
    ce_theta_weight.grad = torch.zeros_like(ce_theta_weight)
    best_theta_weight = ce_theta_weight
    theta_optimizer = torch.optim.Adam([ce_theta_weight], lr=args.sungen_outer_lr)

    # Solve the problem - find best theta for each training sample
    for i in range(args.sungen_outer_epoch):
        write_log(logger, f"############################################################\n\
                                    Outer epoch {i+1}/{args.sungen_outer_epoch}\n\
                       ############################################################\n")

        if args.sungen_use_sigmoid:
            ce_theta_mapped = torch.sigmoid(ce_theta_weight)
        else:
            ce_theta_mapped = ce_theta_weight
        # write_log(logger, f"ce_theta_mapped: {ce_theta_mapped}")

        model_train_diverged = True
        # Start Inner Loop
        while model_train_diverged:
            model_copy_converged, inner_loss, inner_acc, model_weights_cache, opt_checkpoints_cache, model_train_diverged = \
            sungen_inner_train(model, dataloader_dict['train'], ce_theta_mapped.detach(), args, device)
            write_log(logger, f"Diverged: {model_train_diverged} / Inner Loss: {inner_loss} / Inner Acc: {inner_acc}")

        valid_set = build_valid_subset(args, dataset_dict['train'])
        dataloader_dict['valid'] = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True, drop_last=True)

        grad_weights, valid_acc_cls, valid_loss_cls = \
        sungen_outer_get_grad_on_valid(model_copy_converged, dataloader_dict['valid'], ce_theta_mapped.detach(), args, device)
        write_log(logger, f"Valid Loss: {valid_loss_cls} / Valid Acc: {valid_acc_cls}")

        grad_theta = repass_backward(model, model_weights_cache[0], opt_checkpoints_cache[0],
                                     grad_weights, dataloader_dict['train'], ce_theta_mapped, ce_theta_weight, args, device)

        theta_optimizer.zero_grad()
        write_log(logger, f"Sum grads {sum([g for g in grad_theta])}")
        with torch.no_grad():
            ce_theta_weight.grad += grad_theta.data
        torch.nn.utils.clip_grad_norm_(ce_theta_weight, args.clip_grad_norm)
        theta_optimizer.step()

        if not args.sungen_use_sigmoid:
            with torch.no_grad():
                ce_theta_weight.data.clamp_(0, 1)
        torch.cuda.empty_cache()

        # Save best theta
        best_theta_weight = copy.deepcopy(ce_theta_weight)

    # Save best theta
    checkpoint_save_path = os.path.join(args.checkpoint_path, args.task, args.task_dataset, args.model_type)
    check_path(checkpoint_save_path)
    torch.save({
        'theta': best_theta_weight,
    }, os.path.join(checkpoint_save_path, f'{args.training_type}_best_theta.pt'))

    # Visualize theta as a histogram
    plt.hist(best_theta_weight.cpu().detach().numpy(), bins=100)
    plt.savefig(os.path.join('./', f'{args.training_type}_best_theta.png'))
    plt.close()

    print(f"Best theta: {best_theta_weight}")


    return best_theta_weight

def sungen_train(args: argparse.Namespace):
    device = get_torch_device(args.device)

    # Define logger
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Load best theta
    checkpoint_save_path = os.path.join(args.checkpoint_path, args.task, args.task_dataset, args.model_type)
    check_path(checkpoint_save_path)
    if args.training_type in ['unigen', 'unigen_ablation_noisy_label', 'unigen_ablation_hard_label']:
        checkpoint = torch.load(os.path.join(checkpoint_save_path, 'unigen_best_theta.pt'))
    else:
        checkpoint = torch.load(os.path.join(checkpoint_save_path, 'sungen_best_theta.pt'))
    best_theta_weight = checkpoint['theta'].to(device)

    # Load dataset and define dataloader
    write_log(logger, "Loading data")
    dataset_dict, dataloader_dict = {}, {}

    if args.training_type == 'sungen':
        # SunGen shares the same generated data with ZeroGen
        dataset_dict['train'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset,
                                                      f'train_ZG_NL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))
        dataset_dict['valid'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, args.task_dataset,
                                                      f'valid_ZG_NL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))
    elif args.training_type == 'unigen':
        dataset_dict['train'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, # UniGen is not dataset-specific
                                                      f'train_UG_SL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))
        dataset_dict['valid'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, # UniGen is not dataset-specific
                                                      f'valid_UG_SL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))
    elif args.training_type == 'unigen_ablation_noisy_label':
        dataset_dict['train'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, # UniGen is not dataset-specific
                                                      f'train_UG_NL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))
        dataset_dict['valid'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, # UniGen is not dataset-specific
                                                      f'valid_UG_NL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))
    elif args.training_type == 'unigen_ablation_hard_label':
        dataset_dict['train'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, # UniGen is not dataset-specific
                                                      f'train_UG_HL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))
        dataset_dict['valid'] = ClassificationDataset(args, os.path.join(args.preprocess_path, args.task, # UniGen is not dataset-specific
                                                      f'valid_UG_HL_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pkl'))

    dataset_dict['train'] = build_solved_subset(args, dataset_dict['train'], best_theta_weight)

    dataloader_dict['train'] = DataLoader(dataset_dict['train'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=True, pin_memory=True, drop_last=True)
    dataloader_dict['valid'] = DataLoader(dataset_dict['valid'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=False, pin_memory=True, drop_last=False)
    model_name = get_huggingface_model_name(args.model_type)
    config = AutoConfig.from_pretrained(model_name)
    args.vocab_size = config.vocab_size
    args.num_classes = dataset_dict['train'].num_classes

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Train dataset size / iterations: {len(dataset_dict['train'])} / {len(dataloader_dict['train'])}")
    write_log(logger, f"Valid dataset size / iterations: {len(dataset_dict['valid'])} / {len(dataloader_dict['valid'])}")

    # Get model instance
    write_log(logger, "Building model")
    model = ClassificationModel(args).to(device)
    if args.supcon_loss_weight > 0 and args.supcon_momentum_tau > 0:
        # Create a momentum model for MoCo-style training
        momentum_model = ClassificationModel(args).to(device)

        # Momentum model does not need gradient
        for param in momentum_model.parameters():
            param.requires_grad = False

    # Define optimizer and scheduler
    write_log(logger, "Building optimizer and scheduler")
    optimizer = get_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay, optim_type=args.optimizer)
    scheduler = get_scheduler(optimizer, len(dataloader_dict['train']), num_epochs=args.num_epochs,
                              early_stopping_patience=args.early_stopping_patience, learning_rate=args.learning_rate,
                              scheduler_type=args.scheduler, args=args)
    write_log(logger, f"Optimizer: {optimizer}")
    write_log(logger, f"Scheduler: {scheduler}")

    # Define loss function
    # cls_loss = nn.CrossEntropyLoss()
    if args.supcon_loss_weight > 0:
        supcon_loss = SupConLoss(args)

    # Initialize tensorboard writer
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Initialize wandb
    if args.use_wandb and args.job == 'training':
        import wandb # Only import wandb when it is used
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                       name=get_wandb_exp_name(args),
                       config=args,
                       notes=args.description,
                       tags=["TRAIN",
                             f"Dataset: {args.task_dataset}",
                             f"Model: {args.model_type}",
                             f"Training: {args.training_type}"])

    # Train/Valid - Start training
    start_epoch = 0
    best_epoch_idx = 0
    best_valid_objective_value = None
    early_stopping_counter = 0

    write_log(logger, f"Start training from epoch {start_epoch}")
    for epoch_idx in range(start_epoch, args.num_epochs):
        # Train - Set model to train mode
        model = model.train()
        train_loss_cls = 0
        train_loss_supcon = 0
        train_loss_total = 0
        train_acc_cls = 0
        train_f1_cls = 0

        # Train - Iterate one epoch
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['train'], total=len(dataloader_dict['train']), desc=f'Training - Epoch [{epoch_idx}/{args.num_epochs}]', position=0, leave=True)):
            # Train - Get input data
            input_data = {k: v.to(device) for k, v in data_dicts['input_data'].items()}
            labels = data_dicts['label'].to(device)
            soft_labels = data_dicts['soft_label'].to(device)
            index = data_dicts['index'].to(device)

            # Train - Forward
            classification_logits, projected_cls = model(input_ids=input_data['input_ids'],
                                                         attention_mask=input_data['attention_mask'],
                                                         token_type_ids=input_data['token_type_ids'] if args.model_type not in ['distilbert', 'roberta', 'roberta_large'] else None)

            # Train - Calculate loss
            batch_loss_cls = F.cross_entropy(classification_logits, soft_labels, reduction='none').flatten() * best_theta_weight[index]
            batch_loss_cls = batch_loss_cls.mean()
            if args.supcon_loss_weight > 0:
                if args.supcon_momentum_tau > 0:
                    # Use momentum model for MoCo-style training
                    _, moco_cls = momentum_model(input_ids=input_data['input_ids'],
                                                 attention_mask=input_data['attention_mask'],
                                                 token_type_ids=input_data['token_type_ids'] if args.model_type not in ['distilbert', 'roberta', 'roberta_large'] else None)
                else:
                    moco_cls = None
                batch_loss_supcon = supcon_loss(projected_cls, labels, moco_cls=moco_cls, theta_weight=best_theta_weight[index])
            else:
                batch_loss_supcon = torch.tensor(0.0).to(device)

            batch_loss_total = batch_loss_cls + args.supcon_loss_weight * batch_loss_supcon

            batch_acc_cls = (classification_logits.argmax(dim=-1) == labels).float().mean()
            batch_f1_cls = f1_score(labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')

            # Train - Backward pass
            optimizer.zero_grad()
            batch_loss_total.backward()
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if args.scheduler in ['StepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                scheduler.step() # These schedulers require step() after every training iteration
            if args.supcon_loss_weight > 0 and args.supcon_momentum_tau > 0:
                # Update momentum model for MoCo-style training
                with torch.no_grad():
                    for param_q, param_k in zip(model.parameters(), momentum_model.parameters()):
                        param_k.data = param_k.data * args.supcon_momentum_tau + param_q.data * (1. - args.supcon_momentum_tau)

            # Train - Logging
            train_loss_cls += batch_loss_cls.item()
            train_loss_supcon += batch_loss_supcon.item()
            train_loss_total += batch_loss_total.item()
            train_acc_cls += batch_acc_cls.item()
            train_f1_cls += batch_f1_cls

            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['train']) - 1:
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - Loss: {batch_loss_total.item():.4f}")
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - Acc: {batch_acc_cls.item():.4f}")
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - F1: {batch_f1_cls:.4f}")
            if args.use_tensorboard:
                writer.add_scalar('TRAIN/Learning_Rate', optimizer.param_groups[0]['lr'], epoch_idx * len(dataloader_dict['train']) + iter_idx)

        # Train - End of epoch logging
        if args.use_tensorboard:
            writer.add_scalar('TRAIN/Loss_Cls', train_loss_cls / len(dataloader_dict['train']), epoch_idx)
            writer.add_scalar('TRAIN/Loss_SupCon', train_loss_supcon / len(dataloader_dict['train']), epoch_idx)
            writer.add_scalar('TRAIN/Loss_Total', train_loss_total / len(dataloader_dict['train']), epoch_idx)
            writer.add_scalar('TRAIN/Acc', train_acc_cls / len(dataloader_dict['train']), epoch_idx)
            writer.add_scalar('TRAIN/F1', train_f1_cls / len(dataloader_dict['train']), epoch_idx)

        # Valid - Set model to eval mode
        model = model.eval()
        valid_loss_cls = 0
        valid_loss_total = 0
        valid_acc_cls = 0
        valid_f1_cls = 0

        # Valid - Iterate one epoch
        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['valid'], total=len(dataloader_dict['valid']), desc=f'Validating - Epoch [{epoch_idx}/{args.num_epochs}]', position=0, leave=True)):
            # Valid - Get input data
            input_data = {k: v.to(device) for k, v in data_dicts['input_data'].items()}
            labels = data_dicts['label'].to(device)
            soft_labels = data_dicts['soft_label'].to(device)
            index = data_dicts['index'].to(device)

            # Valid - Forward pass
            with torch.no_grad():
                classification_logits, _ = model(input_ids=input_data['input_ids'],
                                                 attention_mask=input_data['attention_mask'],
                                                 token_type_ids=input_data['token_type_ids'] if args.model_type not in ['distilbert', 'roberta', 'roberta_large'] else None)

            # Valid - Calculate loss and accuracy
            # We don't need to calculate supcon loss for validation
            batch_loss_cls = F.cross_entropy(classification_logits, soft_labels, reduction='none').flatten() * best_theta_weight[index]
            batch_loss_cls = batch_loss_cls.mean()
            batch_loss_total = batch_loss_cls
            batch_acc_cls = (classification_logits.argmax(dim=-1) == labels).float().mean()
            batch_f1_cls = f1_score(labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')

            # Valid - Logging
            valid_loss_cls += batch_loss_cls.item()
            valid_loss_total += batch_loss_total.item()
            valid_acc_cls += batch_acc_cls.item()
            valid_f1_cls += batch_f1_cls

            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['valid']) - 1:
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - Loss: {batch_loss_cls.item():.4f}")
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - Acc: {batch_acc_cls.item():.4f}")
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - F1: {batch_f1_cls:.4f}")

        # Valid - Call scheduler
        if args.scheduler == 'LambdaLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_loss_cls)

        # Valid - Check loss & save model
        valid_loss_cls /= len(dataloader_dict['valid'])
        valid_loss_total /= len(dataloader_dict['valid'])
        valid_acc_cls /= len(dataloader_dict['valid'])
        valid_f1_cls /= len(dataloader_dict['valid'])

        if args.optimize_objective == 'loss':
            valid_objective_value = valid_loss_total
            valid_objective_value = -1 * valid_objective_value # Loss is minimized, but we want to maximize the objective value
        elif args.optimize_objective == 'accuracy':
            valid_objective_value = valid_acc_cls
        elif args.optimize_objective == 'f1':
            valid_objective_value = valid_f1_cls
        else:
            raise NotImplementedError

        if best_valid_objective_value is None or valid_objective_value > best_valid_objective_value:
            best_valid_objective_value = valid_objective_value
            best_epoch_idx = epoch_idx
            write_log(logger, f"VALID - Saving checkpoint for best valid {args.optimize_objective}...")
            early_stopping_counter = 0 # Reset early stopping counter

            checkpoint_save_path = os.path.join(args.checkpoint_path, args.task, args.task_dataset, args.model_type)
            check_path(checkpoint_save_path)

            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None
            }, os.path.join(checkpoint_save_path, f'checkpoint_{args.training_type}.pt'))
            write_log(logger, f"VALID - Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
            write_log(logger, f"VALID - Saved checkpoint to {checkpoint_save_path}")
        else:
            early_stopping_counter += 1
            write_log(logger, f"VALID - Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")

        # Valid - End of epoch logging
        if args.use_tensorboard:
            writer.add_scalar('VALID/Loss_Cls', valid_loss_cls, epoch_idx)
            writer.add_scalar('VALID/Loss_Total', valid_loss_total, epoch_idx)
            writer.add_scalar('VALID/Acc', valid_acc_cls, epoch_idx)
            writer.add_scalar('VALID/F1', valid_f1_cls, epoch_idx)
        if args.use_wandb:
            wandb.log({'TRAIN/Epoch_Loss_Cls': train_loss_cls / len(dataloader_dict['train']),
                       'TRAIN/Epoch_Loss_SupCon': train_loss_supcon / len(dataloader_dict['train']),
                       'TRAIN/Epoch_Loss_Total': train_loss_total / len(dataloader_dict['train']),
                       'TRAIN/Epoch_Acc': train_acc_cls / len(dataloader_dict['train']),
                       'TRAIN/Epoch_F1': train_f1_cls / len(dataloader_dict['train']),
                       'VALID/Epoch_Loss_Cls': valid_loss_cls,
                       'VALID/Epoch_Acc': valid_acc_cls,
                       'VALID/Epoch_F1': valid_f1_cls,
                       'Epoch_Index': epoch_idx})
            wandb.alert(
                title='Epoch End',
                text=f"VALID - Epoch {epoch_idx} - Loss: {valid_loss_cls:.4f} - Acc: {valid_acc_cls:.4f}",
                level=AlertLevel.INFO,
                wait_duration=300
            )

        # Valid - Early stopping
        if early_stopping_counter >= args.early_stopping_patience:
            write_log(logger, f"VALID - Early stopping at epoch {epoch_idx}...")
            break

    # Final - End of training
    write_log(logger, f"Done! Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
    if args.use_tensorboard:
        writer.add_text('VALID/Best', f"Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}")
        writer.close()

    # Final - Save best checkpoint as result model
    if args.training_type == 'sungen':
        final_model_save_path = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type)
    elif args.training_type in ['unigen', 'unigen_ablation_noisy_label', 'unigen_ablation_hard_label']:
        final_model_save_path = os.path.join(args.model_path, args.task, args.model_type)
    final_model_save_name = f'final_model_{args.training_type}_{args.gen_amount}_topk{args.gen_top_k}_topp{args.gen_top_p}_temp{args.gen_temperature}_retemp{args.gen_relabel_temperature}_th{args.gen_relabel_threshold}.pt'

    check_path(final_model_save_path)
    shutil.copyfile(os.path.join(checkpoint_save_path, f'checkpoint_{args.training_type}.pt'), os.path.join(final_model_save_path, final_model_save_name)) # Copy best checkpoint as final model
    write_log(logger, f"FINAL - Saved final model to {final_model_save_path}")

    if args.use_wandb:
        wandb.finish()

def sungen_inner_train(model, train_loader, theta, args, device):
    model_copy = copy.deepcopy(model)

    model_optimizer = get_optimizer(model_copy, learning_rate=args.sungen_inner_lr, weight_decay=args.weight_decay, args=args)
    model_weights_cache = []
    opt_checkpoints_cache = []
    model_train_diverged = False
    train_acc_cls = 0
    train_loss_cls = 0
    for epoch_idx in range(0, args.sungen_inner_epoch):
        model_copy = model_copy.train()
        epoch_acc_cls = 0
        epoch_loss_cls = 0
        for iter_idx, data_dicts in enumerate(tqdm(train_loader, total=len(train_loader), desc="Inner Training", position=0, leave=True)):
            # Inner - Get input data
            input_data = {k: v.to(device) for k, v in data_dicts['input_data'].items()}
            labels = data_dicts['label'].to(device)
            soft_labels = data_dicts['soft_label'].to(device)
            index = data_dicts['index'].to(device)

            # Inner - Forward
            model_copy.zero_grad()
            classification_logits, _ = model_copy(input_ids=input_data['input_ids'],
                                                    attention_mask=input_data['attention_mask'],
                                                    token_type_ids=input_data['token_type_ids'] if args.model_type not in ['distilbert', 'roberta', 'roberta_large'] else None)

            # Inner - Calculate loss
            inner_ce_loss = F.cross_entropy(classification_logits, soft_labels, reduction='none').flatten() * theta[index]
            inner_ce_loss = inner_ce_loss.mean()
            # We will not use supcon_loss for solving process
            model_optimizer.zero_grad()
            inner_ce_loss.backward()
            model_optimizer.step()
            # Update accuracy and loss
            epoch_acc_cls += (classification_logits.argmax(dim=-1) == labels).sum().item() / classification_logits.shape[0]
            epoch_loss_cls += inner_ce_loss.item()

        train_loss_cls += epoch_loss_cls / len(train_loader)
        train_acc_cls += epoch_acc_cls / len(train_loader)


    train_loss_cls /= args.sungen_inner_epoch
    train_acc_cls /= args.sungen_inner_epoch

    opt_checkpoints_cache.append(copy.deepcopy(model_optimizer.state_dict()))
    model_weights_cache.append(copy.deepcopy(model_copy.state_dict()))
    if math.isnan(inner_ce_loss.item()):
        model_train_diverged = True

    return model_copy, train_loss_cls, train_acc_cls, model_weights_cache, opt_checkpoints_cache, model_train_diverged

def sungen_outer_get_grad_on_valid(model, valid_loader, theta, args, device):
    # Start Outer Loop
    grad_weights = []
    valid_acc_cls = 0
    valid_loss_cls = 0
    for iter_idx, data_dicts in enumerate(tqdm(valid_loader, total=len(valid_loader), desc="Outer Grad", position=0, leave=True)):
        # Outer - Get input data
        input_data = {k: v.to(device) for k, v in data_dicts['input_data'].items()}
        labels = data_dicts['label'].to(device)
        soft_labels = data_dicts['soft_label'].to(device)
        index = data_dicts['index'].to(device)

        # Outer - Forward
        classification_logits, _ = model(input_ids=input_data['input_ids'],
                                         attention_mask=input_data['attention_mask'],
                                         token_type_ids=input_data['token_type_ids'] if args.model_type not in ['distilbert', 'roberta', 'roberta_large'] else None)

        # Outer - Get theta for the validation batch
        theta_batch = theta[index]
        val_theta, _ = torch.topk(theta, k=int(args.sungen_threshold * len(theta)))
        theta_subnet = (theta_batch >= val_theta[-1]).float()
        theta_selection = torch.nonzero(theta_subnet.squeeze()).flatten()

        # Outer - Calculate loss
        one_hot = torch.zeros(len(labels), args.num_classes).to(device)
        one_hot.scatter_(1, labels.view(-1, 1), args.sungen_init_label).to(device)
        one_hot = F.softmax(one_hot, dim=-1)
        loss_term1 = F.softmax(classification_logits, dim=1)
        loss_term2 = F.log_softmax(classification_logits, dim=1) - torch.log(one_hot + 1e-10) # Add 1e-10 to prevent log(0)
        loss_term3 = torch.mul(loss_term1, F.log_softmax(classification_logits, dim=1))
        valid_loss = loss_term1 * loss_term2 - loss_term3
        valid_loss = torch.mean(valid_loss[theta_selection])

        # Outer - Calculate accuracy
        valid_acc_cls += (classification_logits.argmax(dim=-1) == labels).sum().item() / classification_logits.shape[0]
        valid_loss_cls += valid_loss.item()

        # Outer - Calculate gradient
        grad_weights_batch = torch.autograd.grad(valid_loss, model.parameters())
        if iter_idx > 0:
            grad_weights = [wb+w for wb, w in zip(grad_weights_batch, grad_weights)]
        else:
            grad_weights = grad_weights_batch # Initialize grad_weights
    valid_loss_cls /= len(valid_loader)
    valid_acc_cls /= len(valid_loader)

    return grad_weights, valid_loss_cls, valid_acc_cls

def build_valid_subset(args, train_set: ClassificationDataset) -> ClassificationDataset:
    """
    Build a subset of training dataset for validation, for sungen/unigen training.
    """

    valid_subset = copy.deepcopy(train_set)

    # Randomly sample args.sungen_valid_size data from train_set
    valid_subset.data_list = random.sample(valid_subset.data_list, args.sungen_valid_size)

    return valid_subset

def build_solved_subset(args, train_set: ClassificationDataset, theta: torch.Tensor) -> ClassificationDataset:
    """
    Build a subset of training dataset for validation, for sungen/unigen training.
    """

    solved_subset = copy.deepcopy(train_set)

    # Sort theta in descending order
    theta_sorted, theta_sorted_idx = torch.sort(theta, descending=True)

    # Select top args.sungen_train_size data from train_set
    solved_subset.data_list = [solved_subset.data_list[i] for i in theta_sorted_idx[:args.sungen_train_size]]

    return solved_subset

def repass_backward(model, model_cache, optimizer_cache, grad_weights, dataloader, theta_mapped, theta, args, device):
    # accumulate gradients backwards to leverage hessian-vector product
    theta_grads = [torch.zeros_like(theta)]
    theta_sum = theta_mapped.detach().sum()

    model_copy = copy.deepcopy(model)
    model_copy.load_state_dict(model_cache)

    with torch.backends.cudnn.flags(enabled=False):
        for iter_idx, data_dicts in enumerate(tqdm(dataloader, total=len(dataloader), desc="Repass", position=0, leave=True)):
            # Repass - Get input data
            input_data = {k: v.to(device) for k, v in data_dicts['input_data'].items()}
            # labels = data_dicts['label'].to(device)
            soft_labels = data_dicts['soft_label'].to(device)
            index = data_dicts['index'].to(device)

            # Repass - Forward
            _, w_mapped = pseudo_updated_params(model_copy, optimizer_cache, input_data, soft_labels, theta_mapped[index], theta_sum, args)
            grad_batch = torch.autograd.grad(w_mapped, theta, grad_outputs=grad_weights, retain_graph=True)
            theta_grads = [g+b for g, b in zip(theta_grads, grad_batch)]

    return theta_grads[0]

def pseudo_updated_params(pseudo_model, optimizer_cache, input_data, soft_labels, theta, theta_sum, args):
    pseudo_optimizer = MetaSGD(pseudo_model, pseudo_model.parameters(), lr=args.sungen_inner_lr)
    w_old = [p for p in pseudo_model.parameters()]
    output, _ = pseudo_model(input_ids=input_data['input_ids'],
                             attention_mask=input_data['attention_mask'],
                             token_type_ids=input_data['token_type_ids'] if args.model_type not in ['distilbert', 'roberta', 'roberta_large'] else None)

    pseudo_loss_vector = F.cross_entropy(output, soft_labels, reduction='none').flatten()
    pseudo_loss_vector *= theta
    pseudo_loss = torch.sum(pseudo_loss_vector/theta_sum)
    pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_model.parameters(), create_graph=True)
    w_mapped = pseudo_optimizer.meta_step_adam(pseudo_grads, lr=optimizer_cache['param_groups'][0]['lr'])

    return w_old, w_mapped
