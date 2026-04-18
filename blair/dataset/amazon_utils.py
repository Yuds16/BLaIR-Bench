# blair/dataset/utils.py

import os
import re
import html
import json
from datasets import load_dataset


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def filter_items_wo_metadata(example, item2meta):
    if example['parent_asin'] not in item2meta:
        example['history'] = ''
    history = example['history'].split(' ')
    filtered_history = [_ for _ in history if _ in item2meta]
    example['history'] = ' '.join(filtered_history)
    return example


def truncate_history(example, max_his_len):
    example['history'] = ' '.join(example['history'].split(' ')[-max_his_len:])
    return example


def remap_id(datasets):
    user2id = {'[PAD]': 0}
    id2user = ['[PAD]']
    item2id = {'[PAD]': 0}
    id2item = ['[PAD]']

    # Collect all unique items first
    all_items = set()
    for split in ['train', 'valid', 'test']:
        dataset = datasets[split]
        for item_id, history in zip(dataset['parent_asin'], dataset['history']):
            all_items.add(item_id)
            items_in_history = history.split(' ')
            for item in items_in_history:
                if item:  # Skip empty strings
                    all_items.add(item)
    
    # Sort items for consistent ordering (same as cf would encounter them)
    for item in sorted(all_items):
        item2id[item] = len(id2item)
        id2item.append(item)

    # Process users (order doesn't matter for users since embeddings are item-based)
    for split in ['train', 'valid', 'test']:
        dataset = datasets[split]
        for user_id in dataset['user_id']:
            if user_id not in user2id:
                user2id[user_id] = len(id2user)
                id2user.append(user_id)

    data_maps = {'user2id': user2id, 'id2user': id2user, 'item2id': item2id, 'id2item': id2item}
    return data_maps


def list_to_str(l):
    if isinstance(l, list):
        return list_to_str(', '.join(l))
    else:
        return l


def clean_text(raw_text):
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text


def feature_process(feature):
    sentence = ""
    if isinstance(feature, float):
        sentence += str(feature)
        sentence += '.'
    elif isinstance(feature, list) and len(feature) > 0:
        for v in feature:
            sentence += clean_text(v)
            sentence += ', '
        sentence = sentence[:-2]
        sentence += '.'
    else:
        sentence = clean_text(feature)
    return sentence + ' '


def clean_metadata(example, features_needed:list = ['title']):
    meta_text = ''
    for feature in features_needed:
        meta_text += feature_process(example[feature])
    example['cleaned_metadata'] = meta_text
    return example


def process_meta(domain, n_workers, features_needed=['title']):

    meta_dataset = load_dataset(
        'McAuley-Lab/Amazon-Reviews-2023',
        f'raw_meta_{domain}',
        split='full',
        trust_remote_code=True
    )

    meta_dataset = meta_dataset.map(
        lambda example: clean_metadata(example, features_needed=features_needed),
        num_proc=n_workers
    )

    item2meta = {}
    for parent_asin, cleaned_metadata in zip(meta_dataset['parent_asin'], meta_dataset['cleaned_metadata']):
        item2meta[parent_asin] = cleaned_metadata

    return item2meta
