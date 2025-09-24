from config import ROOT, DATA_DIR
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from src.chemoinformatics.utils import smiles_to_mols
from src.chemoinformatics.descriptors import mols_to_ecfp
from tdc.single_pred import Tox


class MoleculeDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = config['data']['dataset_name']
        self.descriptor = config['data']['descriptor']

    def setup(self, stage=None):

        if self.name == 'PPARy':
            df = pd.read_csv(os.path.join(DATA_DIR, 'CHEMBL3979_EC50.csv'))
            self.smiles = df['smiles']
            self.labels = [1 if i < 100 else 0 for i in df['exp_mean [nM]']]  # 1 if active (<100nM), 0 if inactive (>100nM)

            mols = smiles_to_mols(self.smiles)
            X = mols_to_ecfp(mols, radius=2, nbits=2048, to_array=True)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(self.labels, dtype=torch.long)
        elif self.name == 'Tox':

            df = Tox(name = 'DILI').get_data()
            self.smiles = df['Drug'].to_list()
            self.labels = df['Y']

            mols = smiles_to_mols(self.smiles)
            X = mols_to_ecfp(mols, radius=2, nbits=2048, to_array=True)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(self.labels, dtype=torch.long)

        self.train_ds = TensorDataset(X, y)

    def train_dataloader(self):
        d = self.config['data']
        return DataLoader(self.train_ds,
                          batch_size=d['batch_size'],
                          num_workers=d['num_workers'],
                          shuffle=True,
                          pin_memory=True)
    