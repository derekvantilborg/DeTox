from config import ROOT, DATA_DIR
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from src.chemoinformatics.utils import smiles_to_mols
from src.chemoinformatics.descriptors import mols_to_ecfp, mols_to_cats
from tdc.single_pred import Tox
from tdc.utils import create_scaffold_split


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = config['data']['dataset_name']
        self.descriptor = config['data']['descriptor']
        self.seed = config['data']['splitting_seed']
        self.splitting_ratios = config['data']['splitting_ratios']

    def setup(self, stage=None):

        if self.name == 'PPARy':
            df = pd.read_csv(os.path.join(DATA_DIR, 'CHEMBL3979_EC50.csv'))
        elif self.name == 'Tox':
            df = Tox(name = 'DILI').get_data()

        # Create scaffold splits
        splits = create_scaffold_split(df, seed=1, frac=[0.7, 0.1, 0.2], entity='Drug')

        # Train set
        train_mols = smiles_to_mols(splits['train']['Drug'])
        if self.descriptor == 'ECFP4':
            train_X = mols_to_ecfp(train_mols, radius=2, nbits=2048, to_array=True)
        elif self.descriptor == 'CATs':
            train_X = mols_to_cats(train_mols)
        train_X = torch.tensor(train_X, dtype=torch.float32)
        train_y = torch.tensor(splits['train']['Y'].values, dtype=torch.long)
        self.train_ds = TensorDataset(train_X, train_y)

        # Val set
        val_mols = smiles_to_mols(splits['valid']['Drug'])
        if self.descriptor == 'ECFP4':
            val_X = mols_to_ecfp(val_mols, radius=2, nbits=2048, to_array=True)
        elif self.descriptor == 'CATs':
            val_X = mols_to_cats(val_mols)
        val_X = torch.tensor(val_X, dtype=torch.float32)
        val_y = torch.tensor(splits['valid']['Y'].values, dtype=torch.long)
        self.val_ds = TensorDataset(val_X, val_y)

        # Test set
        test_mols = smiles_to_mols(splits['test']['Drug'])
        if self.descriptor == 'ECFP4':
            test_X = mols_to_ecfp(test_mols, radius=2, nbits=2048, to_array=True)
        elif self.descriptor == 'CATs':
            test_X = mols_to_cats(test_mols)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        test_y = torch.tensor(splits['test']['Y'].values, dtype=torch.long)
        self.test_ds = TensorDataset(test_X, test_y)

    def train_dataloader(self):
        d = self.config['data']
        return DataLoader(self.train_ds,
                          batch_size=d['batch_size'],
                          num_workers=d['num_workers'],
                          shuffle=True,
                          pin_memory=True)
    
    def val_dataloader(self):
        d = self.config['data']
        return DataLoader(self.val_ds,
                        batch_size=d['batch_size'],
                        num_workers=d['num_workers'],
                        shuffle=False,
                        pin_memory=True)

    def test_dataloader(self):
        d = self.config['data']
        return DataLoader(self.test_ds,
                        batch_size=d['batch_size'],
                        num_workers=d['num_workers'],
                        shuffle=False,
                            pin_memory=True)
