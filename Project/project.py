

import numpy as np
import torch, os, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from time import time
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from sklearn.covariance import EllipticEnvelope
from sklearn.utils import shuffle

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
    print('cpu')


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

ano_sample, ano_test = np.load("anomaly_sample.npy"), np.load("anomalytestdata.npy")
traindata, trainlabel = np.load("traindata.npy"), np.load("trainlabel.npy")
testdata = np.load("testdata.npy")

norm=True
if norm:
    print("preprocessing data") 
    scalerA = StandardScaler()
    scalerA.fit(traindata[:,0,:])

    tmp = scalerA.transform(traindata[:,0,:])
    traindata[:,0,:] = tmp
    tmp = scalerA.transform(testdata[:,0,:])
    testdata[:,0,:] = tmp
    tmp = scalerA.transform(ano_sample[:,0,:])
    ano_sample[:,0,:] = tmp
    tmp = scalerA.transform(ano_test[:,0,:])
    ano_test[:,0,:] = tmp
    

    scalerB = StandardScaler()
    scalerB.fit(traindata[:,1,:])
    tmp = scalerB.transform(traindata[:,1,:])
    traindata[:,1,:] = tmp
    tmp = scalerB.transform(testdata[:,1,:])
    testdata[:,1,:] = tmp
    tmp = scalerA.transform(ano_sample[:,1,:])
    ano_sample[:,1,:] = tmp
    tmp = scalerA.transform(ano_test[:,1,:])
    ano_test[:,1,:] = tmp

train0=[]
train1=[]
train2=[]

for i in range(len(traindata)):
  if trainlabel[i]==0:
    train0.append(traindata[i])
  elif trainlabel[i]==1:
    train1.append(traindata[i])
  elif trainlabel[i]==2:
    train2.append(traindata[i])

train0=np.array(train0)
train1=np.array(train1)
train2=np.array(train2)

trainset = RawDataset(traindata,trainlabel) 
trainloader = DataLoader(trainset,batch_size=48,shuffle=True)
testset = RawDataset(ano_test,None) 
testloader = DataLoader(testset,batch_size=1,shuffle=False)

t = RawDataset(testdata,None) 
tloader = DataLoader(t,batch_size=1,shuffle=False)

test0=RawDataset(train0,None)
test0loader = DataLoader(test0,batch_size=1,shuffle=False)
test1=RawDataset(train1,None)
test1loader = DataLoader(test1,batch_size=1,shuffle=False)
test2=RawDataset(train2,None)
test2loader = DataLoader(test2,batch_size=1,shuffle=False)
anoset=RawDataset(ano_sample,None)
anoloader = DataLoader(anoset,batch_size=1,shuffle=False)

class MyDSPNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # define your arch
        self.encoder = nn.Sequential(
            nn.Conv1d(2,20,kernel_size=13,stride=7),
            nn.ReLU(),
            nn.Conv1d(20,40,kernel_size=11,stride=7),
            nn.ReLU(),
            nn.Conv1d(40,80,kernel_size=9,stride=5),
            nn.ReLU(),
            nn.Conv1d(80,160,kernel_size=7,stride=5),
        )
        self.clf = nn.Linear(1920,3)
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
trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=50)
trainer.fit(model, trainloader,)

model.eval()

model = model.to(device)

def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

save_model(model,"ww.pth")

print(device)

test0out=[]
test1out=[]
test2out=[]
anoout=[]
tout=[]
final=[]

with torch.no_grad():
  
      for i, x in enumerate(test0loader):
        x = x.to(device)
        
        test0out.append(model(x).detach().cpu().numpy().reshape(-1)) 
      for i, x in enumerate(test1loader):
        x = x.to(device)
        test1out.append(model(x).detach().cpu().numpy().reshape(-1)) 
      for i, x in enumerate(test2loader):
        x = x.to(device)
        test2out.append(model(x).detach().cpu().numpy().reshape(-1)) 
      for i, x in enumerate(anoloader):
        x = x.to(device)
        anoout.append(model(x).detach().cpu().numpy().reshape(-1)) 
      for i, x in enumerate(tloader):
        x = x.to(device)
        tout.append(model(x).detach().cpu().numpy().reshape(-1)) 
      for i, x in enumerate(testloader):
        x = x.to(device)
        final.append(model(x).detach().cpu().numpy().reshape(-1)) 

test0out=np.array(test0out)
test1out=np.array(test1out)
test2out=np.array(test2out)
anoout=np.array(anoout)
tout=np.array(tout)
final=np.array(final)




trainsvm0=np.concatenate((test0out,test1out[np.random.choice(test1out.shape[0], 150, replace=False)] ,test2out[np.random.choice(test2out.shape[0], 150, replace=False)]),axis=0)
trainsvm1=np.concatenate((test1out,test0out[np.random.choice(test0out.shape[0], 150, replace=False)] ,test2out[np.random.choice(test2out.shape[0], 150, replace=False)]),axis=0)
trainsvm2=np.concatenate((test2out,test1out[np.random.choice(test1out.shape[0], 150, replace=False)] ,test0out[np.random.choice(test0out.shape[0], 150, replace=False)]),axis=0)


clf0=EllipticEnvelope(contamination=0.2)
clf1=EllipticEnvelope(contamination=0.2)
clf2=EllipticEnvelope(contamination=0.2)

clf0.fit(shuffle(trainsvm0))
clf1.fit(shuffle(trainsvm1))
clf2.fit(shuffle(trainsvm2))

y0=clf0.predict(trainsvm1)
y1=clf1.predict(trainsvm1)
y2=clf2.predict(trainsvm0)


print(y0[y0 == 1].size)
print(y1[y1 == 1].size)
print(y2[y2 == 1].size)

def checkano(latent):
  
  if clf0.predict(latent)==-1 and clf1.predict(latent)==-1 and clf2.predict(latent)==-1:
    return True
  return False

with torch.no_grad():
  with open("result.csv","w") as f:
      f.write("id,category\n")
      for i, x in enumerate(testloader):

          if checkano(final[i].reshape(1, -1)) :
            f.write("%d,%d\n"%(i,3))
          else:

            x = x.to(device)
            output = model(x)
            pred = output.argmax(dim=1, keepdim=True)
            f.write("%d,%d\n"%(i,pred.item()))