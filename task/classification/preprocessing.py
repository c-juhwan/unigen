# Standard Library Modules
import os
import sys
import json
import pickle
import argparse
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoConfig
from datasets import load_dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path

def load_data(args: argparse.Namespace) -> tuple: # (dict, dict, dict, int)
    """
    Load data from huggingface datasets.
    If dataset is not in huggingface datasets, takes data from local directory.

    Args:
        dataset_name (str): Dataset name.
        args (argparse.Namespace): Arguments.
        train_valid_split (float): Train-valid split ratio.

    Returns:
        train_data (dict): Training data. (text, label)
        valid_data (dict): Validation data. (text, label)
        test_data (dict): Test data. (text, label)
        num_classes (int): Number of classes.
    """

    name = args.task_dataset.lower()
    train_valid_split = args.train_valid_split

    train_data = {
        'text': [],
        'label': [],
    }
    valid_data = {
        'text': [],
        'label': [],
    }
    test_data = {
        'text': [],
        'label': [],
    }

    if name == 'sst2':
        dataset = load_dataset('gpt3mix/sst2')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()

        # Assign 0 as negative, 1 as positive / Convert 0 -> 1, 1 -> 0
        train_data['label'] = [1 if label == 0 else 0 for label in train_data['label']]
        valid_data['label'] = [1 if label == 0 else 0 for label in valid_data['label']]
        test_data['label'] = [1 if label == 0 else 0 for label in test_data['label']]
    elif name == 'imdb':
        dataset = load_dataset('imdb')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'rotten':
        dataset = load_dataset('rotten_tomatoes')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'cr':
        dataset = load_dataset('SetFit/SentEval-CR')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'yelp_polarity':
        dataset = load_dataset('yelp_polarity')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'amazon_polarity':
        dataset = load_dataset('amazon_polarity')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['content'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['content'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['content'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'tweet_sentiment_binary':
        dataset = load_dataset('tweet_eval', name='sentiment')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # Remove neutral data
        train_df = train_df[train_df['label'] != 1]
        valid_df = valid_df[valid_df['label'] != 1]
        test_df = test_df[test_df['label'] != 1]

        # Convert 2 -> 1
        train_df['label'] = [1 if label == 2 else 0 for label in train_df['label']]
        valid_df['label'] = [1 if label == 2 else 0 for label in valid_df['label']]
        test_df['label'] = [1 if label == 2 else 0 for label in test_df['label']]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif 'multi_domain' in name:
        if name == 'multi_domain_book':
            dataset = pd.read_csv(os.path.join(args.data_path, 'text_classification', 'amazon_review', 'multi_domain', 'amazon_multidomain_books.csv'))
        elif name == 'multi_domain_dvd':
            dataset = pd.read_csv(os.path.join(args.data_path, 'text_classification', 'amazon_review', 'multi_domain', 'amazon_multidomain_dvd.csv'))
        elif name == 'multi_domain_kitchen':
            dataset = pd.read_csv(os.path.join(args.data_path, 'text_classification', 'amazon_review', 'multi_domain', 'amazon_multidomain_kitchen.csv'))
        elif name == 'multi_domain_electronics':
            dataset = pd.read_csv(os.path.join(args.data_path, 'text_classification', 'amazon_review', 'multi_domain', 'amazon_multidomain_electronics.csv'))
        num_classes = 2

        # shuffle dataset
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        # train-test split. 20% -> 400 test data
        train_df = dataset[:-400]
        test_df = dataset[-400:]
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif 'amazon_review' in name:
        if name == 'amazon_review_fashion':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'AMAZON_FASHION_5.json')
        elif name == 'amazon_review_beauty':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'All_Beauty_5.json')
        elif name == 'amazon_review_appliances':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Appliances_5.json')
        elif name == 'amazon_review_arts':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Arts_Crafts_and_Sewing_5.json')
        elif name == 'amazon_review_automotive':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Automotive_5.json')
        elif name == 'amazon_review_books':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Books_5.json')
        elif name == 'amazon_review_cds':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'CDs_and_Vinyl_5.json')
        elif name == 'amazon_review_cellphones':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Cell_Phones_and_Accessories_5.json')
        elif name == 'amazon_review_clothing':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Clothing_Shoes_and_Jewelry_5.json')
        elif name == 'amazon_review_digitalmusic':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Digital_Music_5.json')
        elif name == 'amazon_review_electronics':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Electronics_5.json')
        elif name == 'amazon_review_giftcards':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Gift_Cards_5.json')
        elif name == 'amazon_review_grocery':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Grocery_and_Gourmet_Food_5.json')
        elif name == 'amazon_review_home':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Home_and_Kitchen_5.json')
        elif name == 'amazon_review_industrial':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Industrial_and_Scientific_5.json')
        elif name == 'amazon_review_kindle':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Kindle_Store_5.json')
        elif name == 'amazon_review_luxury':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Luxury_Beauty_5.json')
        elif name == 'amazon_review_magazine':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Magazine_Subscriptions_5.json')
        elif name == 'amazon_review_movies':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Movies_and_TV_5.json')
        elif name == 'amazon_review_musicalinstruments':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Musical_Instruments_5.json')
        elif name == 'amazon_review_officeproducts':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Office_Products_5.json')
        elif name == 'amazon_review_patio':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Patio_Lawn_and_Garden_5.json')
        elif name == 'amazon_review_pet':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Pet_Supplies_5.json')
        elif name == 'amazon_review_primepantry':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Prime_Pantry_5.json')
        elif name == 'amazon_review_software':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Software_5.json')
        elif name == 'amazon_review_sports':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Sports_and_Outdoors_5.json')
        elif name == 'amazon_review_tools':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Tools_and_Home_Improvement_5.json')
        elif name == 'amazon_review_toys':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Toys_and_Games_5.json')
        elif name == 'amazon_review_videogames':
            file_name = os.path.join(args.data_path, 'text_classification', 'amazon_review', '5core', 'Video_Games_5.json')

        with open(file_name) as f:
            dataset = f.readlines()
        num_classes = 2
        processed_list = []

        for each_data in tqdm(dataset, total=len(dataset), desc=f'Loading {name} dataset'):
            # Cast string to dict
            each_data = json.loads(each_data)

            if 'reviewText' not in each_data:
                continue

            text = each_data['reviewText']
            text = text.replace('\n', ' ')
            # multiple spaces to single space
            text = ' '.join(text.split())
            rating = each_data['overall']
            if rating == 3:
                continue # skip neutral ratings
            elif rating > 3:
                label = 1 # positive
            else:
                label = 0 # negative

            processed_list.append({
                'text': text,
                'label': label
            })

        dataset = pd.DataFrame(processed_list)
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        # Use 10% as test data
        test_df = dataset[-int(len(dataset) * 0.1):]
        train_df = dataset[:-int(len(dataset) * 0.1)]
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()

    return train_data, valid_data, test_data, num_classes

def preprocessing(args: argparse.Namespace) -> None:
    """
    Main function for preprocessing.

    Args:
        args (argparse.Namespace): Arguments.
    """

    # Load data
    train_data, valid_data, test_data, num_classes = load_data(args)

    # Preprocessing - Define data_dict
    data_dict = {
        'train': {
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': num_classes,
        },
        'valid': {
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': num_classes,
        },
        'test': {
            'input_text': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': num_classes,
        },
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset)
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in range(len(split_data['label'])):
            # Get text and append to data_dict
            text = split_data['text'][idx]
            data_dict[split]['input_text'].append(text)

            # Get label and append to data_dict
            label = split_data['label'][idx]
            data_dict[split]['labels'].append(label)

            soft_label = [0.0] * num_classes
            if label != -1:
                soft_label[label] = 1.0
            data_dict[split]['soft_labels'].append(soft_label)

        # Save data_dict as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_ORI.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)
