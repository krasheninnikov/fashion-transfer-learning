import numpy as np
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split


def load_data(data_path):
    # There were 24 csv rows throwing an error due to incorrectly formatted productDisplayName
    # We fixed these rows in the csv manually
    styles_df = pd.read_csv(data_path + 'styles.csv')
    styles_df['id'] = styles_df['id'].astype(int)

    styles_ids_set = set(styles_df['id'])
    image_ids_set = {int(f[:-4]) for f in os.listdir(data_path + 'images/')}

    # assert that all images have a corresponding styles_df entry
    assert not image_ids_set.difference(styles_ids_set)

    # remove styles_df rows that do not have a corresponding image
    missing_image_ids = styles_ids_set.difference(image_ids_set)
    # print('Missing image ids: {}'.format(missing_image_ids))
    styles_df = styles_df[~styles_df['id'].isin(missing_image_ids)]

    # remove styles_df rows where year or articleType are nans
    styles_df = styles_df.dropna(subset=['articleType', 'year'])
    styles_df['year'] = styles_df['year'].astype(int)
    return styles_df


def data_split_fashion(styles_df):
    # train / test split: even years in train, odd years in test
    styles_df_training = styles_df[styles_df['year'] % 2 == 0]
    styles_df_test = styles_df[styles_df['year'] % 2 == 1]

    # further split the training data into a dataset with 20 most frequent article types and all other article types
    top20_classes = set(styles_df.groupby(['articleType']).size().nlargest(20).index)
    styles_df_training_top20 = styles_df_training[styles_df_training['articleType'].isin(top20_classes)]
    styles_df_training_others = styles_df_training[~styles_df_training['articleType'].isin(top20_classes)]

    return styles_df_training_top20, styles_df_training_others, styles_df_test


class FashionDataset(Dataset):
    def __init__(self, df, data_path, transform, class2idx):
        self.df = df
        self.data_path = data_path
        self.transform = transform
        self.class2idx = class2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Return an image and its label at the given index
        img_id = self.df.iloc[index]['id']
        img_path = self.data_path + 'images/' + str(img_id) + '.jpg'
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        img_class_id = self.class2idx[self.df.iloc[index]['articleType']]
        return img_tensor, img_class_id


def train_val_split(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=val_split)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_dataloaders(data_path, val_split, transform, dataloader_params):
    styles_df = load_data(data_path)
    styles_df_train_top20, styles_df_train_others, styles_df_test = data_split_fashion(styles_df)
    sorted_class_names = list(styles_df.groupby(['articleType']).size().sort_values(ascending=False).index)
    class2idx = {c: i for i, c in enumerate(sorted_class_names)}

    datasets = {}
    datasets['top20'] = FashionDataset(styles_df_train_top20, data_path, transform, class2idx)
    datasets['top20_train'], datasets['top20_val'] = train_val_split(datasets['top20'], val_split)

    datasets['others'] = FashionDataset(styles_df_train_others, data_path, transform, class2idx)
    datasets['others_train'], datasets['others_val'] = train_val_split(datasets['others'], val_split)

    datasets['test'] = FashionDataset(styles_df_test, data_path, transform, class2idx)

    dataloaders = {k: DataLoader(datasets[k], **dataloader_params) for k in datasets}
    return dataloaders
