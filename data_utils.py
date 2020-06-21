import pandas as pd
import os
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image


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


def train_test_split_fashion(styles_df):
    """train / test split: even years in train, odd years in test"""
    styles_df_train = styles_df[styles_df['year'] % 2 == 0]
    styles_df_test = styles_df[styles_df['year'] % 2 == 1]

    train_classes = set(pd.unique(styles_df_train['articleType']))
    test_classes = set(pd.unique(styles_df_test['articleType']))
    print('Splitting the data into test and training set: even years in train, odd years in test.')
    print('Only {} / {} classes have data samples '
          'both in the training and the test set.'.format(len(train_classes.intersection(test_classes)),
                                                          len(train_classes.union(test_classes))))
    # print('The following classes are present in only the training set or only the test set:')
    # print(test_classes.symmetric_difference(train_classes))
    return styles_df_train, styles_df_test


def train_tune_split_fashion(styles_df_train, sorted_class_names):
    # further split the training data into a dataset with 20 most frequent article types and all other article types
    top20_classes = set(sorted_class_names[:20])
    train_classes = set(styles_df_train.groupby('articleType').size().index)
    print('Further splitting the training set into a dataset with 20 most frequent classes and all other classes.')
    print('The training set does not contain {} '
          'contained in 20 most frequent classes.'.format(top20_classes.difference(train_classes)))
    styles_df_train_top20 = styles_df_train[styles_df_train['articleType'].isin(top20_classes)]
    styles_df_train_others = styles_df_train[~styles_df_train['articleType'].isin(top20_classes)]
    return styles_df_train_top20, styles_df_train_others


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


def get_dataloaders(styles_df_train,
                    styles_df_test,
                    sorted_class_names,
                    data_path,
                    val_split,
                    transform,
                    dataloader_params):
    styles_df_train_top20, styles_df_train_others = train_tune_split_fashion(styles_df_train, sorted_class_names)
    class2idx = {c: i for i, c in enumerate(sorted_class_names)}

    datasets = {}
    datasets['train_all'] = FashionDataset(styles_df_train, data_path, transform, class2idx)
    datasets['train'], datasets['val'] = train_val_split(datasets['train_all'], val_split)

    datasets['top20'] = FashionDataset(styles_df_train_top20, data_path, transform, class2idx)
    datasets['train_top20'], datasets['val_top20'] = train_val_split(datasets['top20'], val_split)

    datasets['others'] = FashionDataset(styles_df_train_others, data_path, transform, class2idx)
    datasets['train_others'], datasets['val_others'] = train_val_split(datasets['others'], val_split)

    datasets['test'] = FashionDataset(styles_df_test, data_path, transform, class2idx)

    dataloaders = {k: DataLoader(datasets[k], **dataloader_params) for k in datasets}
    return dataloaders
