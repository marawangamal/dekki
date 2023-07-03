"""
Script for extracting data from Anki .apkg files and saving it as a single integer vector
Example usage:
    python build_data.py  --path /path/to/folder/with/apkg/files
    python build_data.py  --path downloads/anki

"""

import logging
import os
import shutil
import sys
import os.path as osp
import argparse as argparse
import pysqlite3 as sqlite3
import pandas as pd
import zipfile
import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = {
    "raw": osp.join(ROOT_DIR, "data", "raw"),
    "processed": osp.join(ROOT_DIR, "data", "processed"),
    "downloads": osp.join(ROOT_DIR, "data", "downloads"),
}


def anki_make_dataset(path, **kwargs):
    """ Convert .apkg files to tensors in .npy format

    Args:
        path (str): path/to/folder/with/apkg/files
    """

    logger.info("Processing Anki dataset...")
    apkg_zipfiles = [f for f in os.listdir(path) if f.endswith('.zip') or f.endswith('.apkg')]

    # Extract zipfiles - data/downloads/anki/.. ==> data/raw/anki/..
    for apkg_zipfile in apkg_zipfiles:
        logging.info("Loading from {}".format(apkg_zipfile))
        source_path = osp.join(path, apkg_zipfile)
        target_path = osp.join(data_path['raw'], 'anki', apkg_zipfile.split('.')[0])

        if not osp.exists(osp.join(data_path['raw'], 'anki')):
            os.makedirs(osp.join(data_path['raw'], 'anki'))

        # Delete existing folder at data/raw/anki/..
        if osp.exists(target_path):
            logging.info("Deleting existing folder {}".format(target_path))
            shutil.rmtree(target_path)

        # Unzip and move to data/raw/anki/..
        if not osp.exists(osp.join(data_path['raw'], 'anki', apkg_zipfile.split('.')[0])):
            logging.info("Unzipping {}".format(apkg_zipfile))
            with zipfile.ZipFile(source_path, 'r') as zip_ref:
                zip_ref.extractall(target_path)

    # sql to npy: data/raw/anki/.. ==> data/processed/anki/..
    for apkg_folder in os.listdir(osp.join(data_path['raw'], 'anki')):
        # if not apkg_folder in ['luke']:
        #     continue
        db_path = osp.join(data_path['raw'], 'anki', apkg_folder, 'collection.anki21')
        logging.info("Extracting sql data from {}".format(db_path))

        # Delete existing folder at data/processed/anki/..
        if osp.exists(osp.join(data_path['processed'], 'anki', apkg_folder)):
            logging.info("Deleting existing folder {}".format(osp.join(data_path['processed'], 'anki', apkg_folder)))
            shutil.rmtree(osp.join(data_path['processed'], 'anki', apkg_folder))

        # Create folder at data/processed/anki/..
        try:
            extract_data(db_path, osp.join(data_path['processed'], 'anki', apkg_folder))
        except Exception as e:
            logging.error("Failed to extract data from {} with error {}".format(db_path, e))

    # Standardize data
    # import pdb; pdb.set_trace()
    pts = []
    indices_to_standardize = [0, 1, 2]
    for apkg_folder in os.listdir(osp.join(data_path['processed'], 'anki')):
        train_data = np.load(osp.join(data_path['processed'], 'anki', apkg_folder, 'train_data.npy'))  # [B, T, D]
        train_mask = np.load(osp.join(data_path['processed'], 'anki', apkg_folder, 'train_mask.npy'))  # [B, T]

        train_data = train_data[train_mask == 1]

        for i in indices_to_standardize:
            if len(pts) <= i:
                pts.append([])
            pts[i].append(train_data[:, i])

    # import pdb; pdb.set_trace()

    # Concatenate all data
    pts = [np.concatenate(pts[i]) for i in indices_to_standardize]
    means = [np.mean(pts[i]) for i in indices_to_standardize]
    stds = [np.std(pts[i]) for i in indices_to_standardize]

    print("Means: {}".format(means))
    print("Stds: {}".format(stds))

    # Apply standardization
    for apkg_folder in os.listdir(osp.join(data_path['processed'], 'anki')):
        train_data = np.load(osp.join(data_path['processed'], 'anki', apkg_folder, 'train_data.npy'))  # [B, T, D]
        test_data = np.load(osp.join(data_path['processed'], 'anki', apkg_folder, 'test_data.npy'))  # [B, T, D]

        train_mask = np.load(osp.join(data_path['processed'], 'anki', apkg_folder, 'train_mask.npy'))  # [B, T]
        test_mask = np.load(osp.join(data_path['processed'], 'anki', apkg_folder, 'test_mask.npy'))  # [B, T]

        # Standardizing the selected features
        for i in indices_to_standardize:
            train_data[train_mask == 1, i] = (train_data[train_mask == 1, i] - means[i]) / stds[i]
            test_data[test_mask == 1, i] = (test_data[test_mask == 1, i] - means[i]) / stds[i]

        # Save the standardized data
        np.save(osp.join(data_path['processed'], 'anki', apkg_folder, 'train_data.npy'), train_data)
        np.save(osp.join(data_path['processed'], 'anki', apkg_folder, 'test_data.npy'), test_data)

    # import pdb;
    # pdb.set_trace()

def extract_data(source_path, target_path, means=None, stds=None):
    conn = sqlite3.connect(source_path, isolation_level=None, detect_types=sqlite3.PARSE_COLNAMES)
    qry_card_ids = """
        SELECT cards.id
        FROM cards
    """
    cards_timeseries = []
    cards_ids = pd.read_sql_query(qry_card_ids, conn).to_numpy()

    for i, card_id in enumerate(cards_ids):
        qry_note = f"""
            SELECT revlog.id, revlog.cid, revlog.time, revlog.lastIvl, notes.flds, (revlog.ease - 1) as ease
            FROM revlog
            JOIN cards  ON revlog.cid = cards.id
            JOIN notes ON cards.nid = notes.id
            WHERE revlog.cid = {card_id.item()}
        """

        card_ts_df = pd.read_sql_query(qry_note, conn)

        # Get note length
        card_ts_df['note_length'] = card_ts_df['flds'].apply(lambda x: len(x))

        # Sort and filter columns
        card_ts_df = card_ts_df.sort_values(by='id')
        card_ts_df = card_ts_df[['time', 'lastIvl', 'note_length', 'ease']]

        # Convert to numpy
        card_ts = card_ts_df.to_numpy()  # (n, 4)

        if card_ts.shape[0] > 1 and card_ts[:, -1].min() >= 0:
            cards_timeseries.append(card_ts)

        if i % 1000 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(cards_ids)} cards")

    # Create Train / Test split
    train_split = int(len(cards_timeseries) * 0.8)
    train_cards_timeseries = cards_timeseries[:train_split]
    test_cards_timeseries = cards_timeseries[train_split:]

    # Apply padding to all cards and get padding mask
    train_cards_timeseries = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(card_ts) for card_ts in train_cards_timeseries], batch_first=True, padding_value=-100
    ).float()  # (n, max_len, 4)

    test_cards_timeseries = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(card_ts) for card_ts in test_cards_timeseries], batch_first=True, padding_value=-100
    ).float()

    # Get ease counts
    ease_counts = [0, 0, 0, 0]
    for i in range(4):
        ease_counts[i] = torch.sum(train_cards_timeseries[:, :, -1] == i).item()

    # Convert ease to one-hot
    train_cards_timeseries[:, :, -1][train_cards_timeseries[:, :, -1] == -100] = 4
    test_cards_timeseries[:, :, -1][test_cards_timeseries[:, :, -1] == -100] = 4

    train_cards_timeseries = torch.cat(
        (train_cards_timeseries[:, :, :-1],  # (n, max_len, 3)
         torch.nn.functional.one_hot(train_cards_timeseries[:, :, -1].long(), 5).float()),  # (n, max_len, 5)
        dim=-1
    )
    test_cards_timeseries = torch.cat(
        (test_cards_timeseries[:, :, :-1],  # (n, max_len, 3)
         torch.nn.functional.one_hot(test_cards_timeseries[:, :, -1].long(), 5).float()),  # (n, max_len, 5)
        dim=-1
    )

    # Get mask
    train_mask = (train_cards_timeseries[:, :, -1] == 0) * 1
    test_mask = (test_cards_timeseries[:, :, -1] == 0) * 1

    # Save to file (as numpy array)
    if not osp.exists(target_path):
        os.makedirs(target_path)

    np.save(osp.join(target_path, 'train_data.npy'), train_cards_timeseries.numpy())
    np.save(osp.join(target_path, 'train_mask.npy'), train_mask.numpy())
    np.save(osp.join(target_path, 'test_data.npy'), test_cards_timeseries.numpy())
    np.save(osp.join(target_path, 'test_mask.npy'), test_mask.numpy())

    logging.info("Files saved to {}\n\tTrain: {} | Test: {}\n".format(
        osp.join(target_path),
        train_cards_timeseries.shape,
        test_cards_timeseries.shape
    ))

    # Print ease counts
    logging.info(f"Ease counts: {ease_counts}")
    # >> Ease counts: [tensor(38105), tensor(20023), tensor(239857), tensor(27748)]

    return means, stds


make_dataset_functions = {
    "anki": anki_make_dataset,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build datasets for language modelling')
    parser.add_argument('-d', '--dataset', type=str, default='anki', choices=make_dataset_functions.keys())
    parser.add_argument('-p', '--path', type=str, default=osp.join(data_path['downloads'], 'anki'))
    args = parser.parse_args()

    make_dataset_functions[args.dataset](**vars(args))
