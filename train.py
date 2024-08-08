from PIL import Image
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
import lightning as L
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from torch.nn import Sigmoid
#==========
import sys
#sys.path.append('.')
from depth_anything_v2.dpt import DepthAnythingV2
from dataset import STDataset
import ipdb
from torch.optim import AdamW
import matplotlib.pyplot as plt



class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    #def forward(self, pred, target, valid_mask):
    #    valid_mask = valid_mask.detach()
    #    diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
    #    loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
    #                      self.lambd * torch.pow(diff_log.mean(), 2))

    #    return loss
    def forward(self, pred, target):
        ipdb.set_trace()
        diff_log = torch.log(target) - torch.log(pred)
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss



def main():
    debug=False

    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=3)
    loss_checkpoint_callback = ModelCheckpoint(
        verbose=True,
        filename=f"val_loss-" + "epoch_{epoch}-val_loss_{val/loss:.4f}-score_{score/valid_f1:.4f}",
        monitor="valid/loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )
    score_checkpoint_callback = ModelCheckpoint(
        verbose=True,
        filename=f"val_score-" + "epoch_{epoch}-val_loss_{val/loss:.4f}-socre_{score/valid_f1:.4f}",
        monitor="score/train_f1",
        save_top_k=5,
        save_weights_only=True,
        mode="max",
        auto_insert_metric_name=False,
    )

    if not debug:
        logger = WandbLogger(project="demo", name="duck1")
    else:
        logger = TensorBoardLogger("demo", name="duck1")
        
    trainer = L.Trainer(max_epochs=400, max_steps=ep*len(train_dataloader),precision="bf16-mixed", logger=logger, callbacks=[lr_monitor,loss_checkpoint_callback,score_checkpoint_callback],log_every_n_steps=10,accumulate_grad_batches=1,gradient_clip_val=1)
    
    trainer.fit(model=wrapper_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)


def tmp():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**{**model_configs['vits']})
    model.train()
    model=model.cuda()


    #fix dino embedding
    #for name,param in model.named_parameters():
    #    if 'pretrained' in name:
    #        param.requires_grad=False

    #criterion = SiLogLoss().cuda()
    #criterion = nn.MSELoss().cuda()
    criterion = nn.L1Loss().cuda()
    optimizer = AdamW(params=model.parameters(),lr=3e-4)


    #dataset
    st_dataset=STDataset(data_path='img/clean_left/',label_path='img/clean_right/')
    dataloader=DataLoader(dataset=st_dataset,num_workers=20,batch_size=12,shuffle=True)
    sigmoid=Sigmoid()
    k=1.
    while True:
        for i in dataloader:
            optimizer.zero_grad()

            data=i['data'].cuda()
            label=i['label'].cuda()
            out=model(data)
            out=sigmoid(out)
            loss=criterion.forward(input=out,target=label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            print(loss)

            #---test
            k+=1.
            if k%1e3==0:
                out= out[0].detach().cpu().permute(1, 2, 0).numpy() * 255
                out=out.astype(int)
                plt.figure()
                plt.imshow(out)
                plt.savefig(f'{k}.jpg')
            if k%1e4==0:
                torch.save(model.state_dict(), f'model_{k}.pth')


            #out= out[0].detach().cpu().permute(1, 2, 0).numpy() * 255
            #out=out.astype(int)
            #plt.figure()
            #plt.imshow(out)
            #plt.savefig(f'tmp.jpg')


if __name__ == "__main__":
    #seed_everything(42)
    #main()
    tmp()
