import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class MLP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Save the entire config so it’s in checkpoints & logs
        # You can access as self.hparams['model']['hidden_dim'], etc.
        self.save_hyperparameters(config)

        m = self.hparams['model']
        self.net = nn.Sequential(
            nn.Linear(m['input_dim'], m['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(m['dropout']),
            nn.Linear(m['hidden_dim'], m['output_dim'])
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, y)
        self.log("val/loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self(x)
        
        return logits

    def configure_optimizers(self):
        opt_config = self.hparams['optimizer']
        opt = torch.optim.Adam(self.parameters(),
                                lr=opt_config['lr'],
                                weight_decay=opt_config['weight_decay'])

        sch_config = self.hparams.get('scheduler', None)
        if not sch_config or sch_config['name'] is None:
            return opt
      
        return opt
