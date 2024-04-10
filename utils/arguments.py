# Standard Library Modules
import os
import argparse
# Custom Modules
from utils.utils import parse_bool

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.user_name = os.getlogin()
        self.proj_name = 'UniGen'

        # Task arguments
        task_list = ['classification']
        self.parser.add_argument('--task', type=str, choices=task_list, default='classification',
                                 help='Task to do; Must be given.')
        job_list = ['preprocessing', 'training', 'resume_training', 'testing', 'generating', 'inference', 'sungen_solve', 'visualize_tsne']
        self.parser.add_argument('--job', type=str, choices=job_list, default='training',
                                 help='Job to do; Must be given.')
        dataset_list = ['sst2', 'imdb', 'rotten', 'cr', 'yelp_polarity', 'amazon_polarity', 'tweet_sentiment_binary',
                        'multi_domain_book', 'multi_domain_dvd', 'multi_domain_kitchen', 'multi_domain_electronics',
                        'amazon_review_fashion', 'amazon_review_beauty', 'amazon_review_appliances',
                        'amazon_review_arts', 'amazon_review_automotive', 'amazon_review_books',
                        'amazon_review_cds', 'amazon_review_cellphones', 'amazon_review_clothing',
                        'amazon_review_digitalmusic', 'amazon_review_electronics', 'amazon_review_giftcards',
                        'amazon_review_grocery', 'amazon_review_home', 'amazon_review_industrial',
                        'amazon_review_kindle', 'amazon_review_luxury', 'amazon_review_magazines',
                        'amazon_review_movies', 'amazon_review_musicalinstruments', 'amazon_review_officeproducts',
                        'amazon_review_patio', 'amazon_review_pet', 'amazon_review_primepantry',
                        'amazon_review_software', 'amazon_review_sports', 'amazon_review_tools',
                        'amazon_review_toys', 'amazon_review_videogames']
        self.parser.add_argument('--task_dataset', type=str, choices=dataset_list, default='sst2',
                                 help='Dataset for the task; Must be given.')
        self.parser.add_argument('--test_dataset', type=str, choices=dataset_list, default='sst2',
                                 help='Dataset for the test; Must be given for test.')
        training_type_list = ['supervised', 'prompting', 'zerogen', 'sungen', 'unigen', 'zs_inference',
                              'unigen_ablation_noisy_label', 'unigen_ablation_hard_label', 'zerogen_combined']
        self.parser.add_argument('--training_type', type=str, choices=training_type_list, default='supervised',
                                 help='Type of training to use; Default is supervised')
        self.parser.add_argument('--description', type=str, default='default',
                                 help='Description of the experiment; Default is "default"')

        # Path arguments - Modify these paths to fit your environment
        self.parser.add_argument('--data_path', type=str, default=f'/nas_homes/{self.user_name}/dataset',
                                 help='Path to the dataset.')
        self.parser.add_argument('--preprocess_path', type=str, default=f'/nas_homes/{self.user_name}/preprocessed/{self.proj_name}',
                                 help='Path to the preprocessed dataset.')
        self.parser.add_argument('--model_path', type=str, default=f'/nas_homes/{self.user_name}/model_final/{self.proj_name}',
                                 help='Path to the model after training.')
        self.parser.add_argument('--checkpoint_path', type=str, default=f'/nas_homes/{self.user_name}/model_checkpoint/{self.proj_name}')
        self.parser.add_argument('--result_path', type=str, default=f'./results/{self.proj_name}',
                                 help='Path to the result after testing.')
        self.parser.add_argument('--log_path', type=str, default=f'/nas_homes/{self.user_name}/tensorboard_log/{self.proj_name}',
                                 help='Path to the tensorboard log file.')
        self.parser.add_argument('--cls_prompt', type=str, default=f'cls_p1',
                                 help='prompt file for classification.')
        self.parser.add_argument('--gen_prompt', type=str, default=f'gen_p1',
                                 help='prompt file for generation.')

        # Model - Basic arguments
        self.parser.add_argument('--proj_name', type=str, default=self.proj_name,
                                 help='Name of the project.')
        model_type_list = ['bert', 'roberta', 'roberta_large', 'distilbert', 'tinybert', 'lstm', 'cnn', # Classification Model
                           'gpt2', 'gpt2_large', 'gpt2_xl', 'opt', 'bloom'] # Generation Model
        self.parser.add_argument('--model_type', type=str, choices=model_type_list, default='distilbert',
                                 help='Type of the classification model to use.')
        self.parser.add_argument('--model_ispretrained', type=parse_bool, default=True,
                                 help='Whether to use pretrained model; Default is True')
        self.parser.add_argument('--rnn_isbidirectional', type=parse_bool, default=True,
                                 help='Whether to use bidirectional RNNs; Default is True')
        self.parser.add_argument('--min_seq_len', type=int, default=2,
                                 help='Minimum sequence length of the input; Default is 2')
        self.parser.add_argument('--max_seq_len', type=int, default=40,
                                 help='Maximum sequence length of the input; Default is 40')
        self.parser.add_argument('--dropout_rate', type=float, default=0.0,
                                 help='Dropout rate of the model; Default is 0.0')

        # Model - Size arguments
        self.parser.add_argument('--embed_size', type=int, default=768, # Will be automatically specified by the model type if model is PLM
                                 help='Embedding size of the model; Default is 768')
        self.parser.add_argument('--hidden_size', type=int, default=512, # Will be automatically specified by the model type if model is PLM
                                 help='Hidden size of the model; Default is 512')
        self.parser.add_argument('--projection_size', type=int, default=256,
                                 help='Projection size of the model; Default is 256')
        self.parser.add_argument('--num_layers_rnn', type=int, default=1,
                                 help='Number of layers of RNNs; Default is 1')

        # Model - Optimizer & Scheduler arguments
        optim_list = ['SGD', 'AdaDelta', 'Adam', 'AdamW']
        scheduler_list = ['None', 'StepLR', 'LambdaLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau']
        self.parser.add_argument('--optimizer', type=str, choices=optim_list, default='Adam',
                                 help="Optimizer to use; Default is Adam")
        self.parser.add_argument('--scheduler', type=str, choices=scheduler_list, default='None',
                                 help="Scheduler to use for classification; If None, no scheduler is used; Default is None")

        # Training arguments 1
        self.parser.add_argument('--num_epochs', type=int, default=3,
                                 help='Training epochs; Default is 3')
        self.parser.add_argument('--learning_rate', type=float, default=2e-5,
                                 help='Learning rate of optimizer; Default is 2e-5')
        # Training arguments 2
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='Num CPU Workers; Default is 2')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='Batch size; Default is 32')
        self.parser.add_argument('--weight_decay', type=float, default=0,
                                 help='Weight decay; Default is 0; If 0, no weight decay')
        self.parser.add_argument('--clip_grad_norm', type=int, default=3,
                                 help='Gradient clipping norm; Default is 3')
        self.parser.add_argument('--early_stopping_patience', type=int, default=10,
                                 help='Early stopping patience; No early stopping if None; Default is None')
        self.parser.add_argument('--train_valid_split', type=float, default=0.1,
                                 help='Train/Valid split ratio; Default is 0.1')
        objective_list = ['loss', 'accuracy', 'f1']
        self.parser.add_argument('--optimize_objective', type=str, choices=objective_list, default='accuracy',
                                 help='Objective to optimize; Default is accuracy')
        # Training arguments 3 - Supervised Contrastive Learning
        self.parser.add_argument('--supcon_loss_weight', type=float, default=0.5,
                                 help='Weight of the supervised contrastive loss; Default is 0.5')
        self.parser.add_argument('--supcon_temperature', type=float, default=0.2,
                                 help='Temperature of the supervised contrastive loss; Default is 0.2')
        self.parser.add_argument('--supcon_memory_bank_size', type=int, default=64,
                                 help='Memory bank size for supervised contrastive learning; Default is 64')
        self.parser.add_argument('--supcon_momentum_tau', type=float, default=0.999,
                                 help='Momentum tau for supervised contrastive learning; Default is 0.999')
        self.parser.add_argument('--supcon_memory_bank_threshold', type=float, default=0.8,
                                 help='Threshold for theta_weight for supervised contrastive learning; Default is 0.8')

        # UniGen arguments - Generation
        generation_type_list = ['zerogen', 'unigen']
        self.parser.add_argument('--generation_type', type=str, choices=generation_type_list, default='unigen',
                                 help='Type of generation to use; Default is unigen')
        self.parser.add_argument('--gen_top_k', type=int, default=40,
                                 help='k value for top-k sampling (set to 0 to perform no top-k sampling)')
        self.parser.add_argument('--gen_top_p', type=float, default=0.9,
                                 help='p value for top-p sampling (set to 0 to perform no top-p sampling)')
        self.parser.add_argument("--gen_temperature", type=float, default=1.0,
                                 help="The value used to module the next token probabilities.")
        self.parser.add_argument("--gen_amount", type=int, default=1000000,
                                 help="The amount of generated sentences.")
        labeling_type_list = ['none', 'soft', 'hard']
        self.parser.add_argument("--gen_relabel", type=str, choices=labeling_type_list, default='soft',
                                 help="Whether to relabel the generated sentences.")
        self.parser.add_argument("--gen_relabel_temperature", type=float, default=0.1,
                                 help="The temperature value used to balance the relabeling output.")
        self.parser.add_argument("--gen_relabel_threshold", type=float, default=0.2,
                                    help="The threshold value used to relabel the generated sentences.")
        # UniGen arguments - SunGen
        self.parser.add_argument("--sungen_valid_size", type=int, default=50000,
                                 help="The size of validation set for UniGen/SunGen.")
        self.parser.add_argument("--sungen_train_size", type=int, default=200000,
                                 help="The size of training set for UniGen/SunGen.")
        self.parser.add_argument("--sungen_initial_weight", type=float, default=1.0,
                                 help="The initial weight for UniGen/SunGen.")
        self.parser.add_argument("--sungen_outer_lr", type=float, default=1e-1,
                                 help="The outer learning rate for UniGen/SunGen.")
        self.parser.add_argument("--sungen_inner_lr", type=float, default=2e-5,
                                 help="The inner learning rate for UniGen/SunGen.")
        self.parser.add_argument("--sungen_outer_epoch", type=int, default=50,
                                 help="The outer epoch for UniGen/SunGen.")
        self.parser.add_argument("--sungen_inner_epoch", type=int, default=1,
                                 help="The inner epoch for UniGen/SunGen.")
        self.parser.add_argument("--sungen_use_sigmoid", type=parse_bool, default=True,
                                 help="Whether to use sigmoid for theta in UniGen/SunGen.")
        self.parser.add_argument("--sungen_threshold", type=float, default=0.9,
                                 help="The threshold for UniGen/SunGen.")
        self.parser.add_argument("--sungen_init_label", type=int, default=10,
                                 help="The initial label for UniGen/SunGen.")

        # Testing/Inference arguments
        self.parser.add_argument('--test_batch_size', default=32, type=int,
                                 help='Batch size for test; Default is 32')

        # Other arguments - Device, Seed, Logging, etc.
        self.parser.add_argument('--device', type=str, default='cuda',
                                 help='Device to use for training; Default is cuda')
        self.parser.add_argument('--seed', type=int, default=None,
                                 help='Random seed; Default is None')
        self.parser.add_argument('--use_tensorboard', type=parse_bool, default=True,
                                 help='Using tensorboard; Default is True')
        self.parser.add_argument('--use_wandb', type=parse_bool, default=True,
                                 help='Using wandb; Default is True')
        self.parser.add_argument('--log_freq', default=500, type=int,
                                 help='Logging frequency; Default is 500')

    def get_args(self):
        return self.parser.parse_args()
