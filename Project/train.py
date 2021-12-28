import numpy as np
import torch, os, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from time import time
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
import argparse



parser=argparse.ArgumentParser()
parser.add_argument('-data','--data',type=str)
parser.add_argument('-label','--label',type=str)
args=parser.parse_args()


start = time()
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

if torch.cuda.is_available() == True:
    device = "cuda"
    print("Use GPU!")
else:
    device = "cpu"






norm = True
traindata, trainlabel = np.load(args.data), np.load(args.label)

if norm:
    print("preprocessing data")
    scalerA = StandardScaler()
    scalerA.fit(traindata[:,0,:])
    tmp = scalerA.transform(traindata[:,0,:])
    traindata[:,0,:] = tmp
    
    scalerB = StandardScaler()
    scalerB.fit(traindata[:,1,:])
    tmp = scalerB.transform(traindata[:,1,:])
    traindata[:,1,:] = tmp
    

# define dataset here
class RawDataset(Dataset):
    def __init__(self, traindata, trainlabel):
        self.traindata = torch.from_numpy(np.array(traindata).astype(np.float32))
        if trainlabel is not None:
            self.trainlabel = torch.from_numpy(np.array(trainlabel).astype(np.int64))
        else:
            self.trainlabel = None
    def __len__(self):
        return len(self.traindata)
    def __getitem__(self,idx):
        sample = self.traindata[idx]
        if self.trainlabel is not None:
            target = self.trainlabel[idx]
            return sample, target
        else:
            return sample

trainset = RawDataset(traindata,trainlabel) 
trainloader = DataLoader(trainset,batch_size=24,shuffle=True)

class MyDSPNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # define your arch
        self.encoder = nn.Sequential(
            nn.Conv1d(2,32,kernel_size=13,stride=7),
            nn.ReLU(),
            nn.Conv1d(32,64,kernel_size=11,stride=7),
            nn.ReLU(),
            nn.Conv1d(64,128,kernel_size=9,stride=5),
            nn.ReLU(),
            nn.Conv1d(128,256,kernel_size=7,stride=5),
        )
        self.clf = nn.Linear(3072,3)
        
    def forward(self, x):
        # define your forward

        x=self.encoder(x)
        x=x.view(x.size(0),-1)
        output=self.clf(x)
      
        return output
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
        return optimizer
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        self.log('train_loss', loss)
        return loss

# model
model = MyDSPNet()

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=45)
trainer.fit(model, trainloader,)

model = model.to(device)


def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

save_model(model,"weight.pt")



