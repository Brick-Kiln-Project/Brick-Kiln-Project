# try the imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import pathlib
import importlib.util
import torch
import math
import numpy as np
import pandas as pd
import random
import cv2

sys.path.append("../Configs/")
import help_texts
"""
import constants
"""
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torch.nn as nn

class Combo_BI_Dataset(Dataset):
    """
    Dataset for country/ies DL Sentinel2 10-channel imagery.
    """
    def __init__(self, DS_Name,constants, balance=False, limit=0, mode="full",transform=None,device='cuda'):
        super(Combo_BI_Dataset, self).__init__()
        self.df=pd.DataFrame()
        self.upsampled=True
        self.transform=transform
        self.device=device
        
        def add_dataset(dataset_name):
            df=pd.read_csv(constants.IMAGE_DATASETS_ROOT+dataset_name+'/metadata.csv')
            if dataset_name == 'bangladesh_dl_dataset' or dataset_name == 'india_dl_dataset':
                self.upsampled=False
                
            if balance:
                confirmed={'Image':{},'Label':{},'Geometry':{}}
                denied={'Image':{},'Label':{},'Geometry':{}}
                for idx in df['Label'].keys():
                    if df['Label'][idx]:
                        confirmed['Image'][idx]=df['Image'][idx]
                        confirmed['Label'][idx]=df['Label'][idx]
                        confirmed['Geometry'][idx]=df['Geometry'][idx]
                    else:
                        denied['Image'][idx]=df['Image'][idx]
                        denied['Label'][idx]=df['Label'][idx]
                        denied['Geometry'][idx]=df['Geometry'][idx]
                        
                if limit:
                    random.seed(0)
                    sample_keys=random.sample(list(confirmed['Image'].keys()),min([len(confirmed['Image']),limit]))
                    confirmed={'Image':{},'Label':{},'Geometry':{}}
                    for idx in sample_keys:
                        confirmed['Image'][idx]=df['Image'][idx]
                        confirmed['Label'][idx]=df['Label'][idx]
                        confirmed['Geometry'][idx]=df['Geometry'][idx]
                random.seed(0)
                sample_keys=random.sample(list(denied['Image'].keys()),min([len(denied['Image']),math.ceil((len(confirmed['Image'])*3))]))
                denied={'Image':{},'Label':{},'Geometry':{}}
                for idx in sample_keys:
                    denied['Image'][idx]=df['Image'][idx]
                    denied['Label'][idx]=df['Label'][idx]
                    denied['Geometry'][idx]=df['Geometry'][idx]
                confirmed['Image'].update(denied['Image'])
                confirmed['Label'].update(denied['Label'])
                confirmed['Geometry'].update(denied['Geometry'])
                df=pd.DataFrame().from_dict(confirmed)
            #"""
            df["Image"]=constants.IMAGE_DATASETS_ROOT+dataset_name+'/'+df["Image"]
            df=df[['Image','Label']]
            return df
            
    # ADD DATASETS HERE FOR TRAINING
        
        self.df = pd.concat([self.df,add_dataset(DS_Name)])
        
    # END EDITABLES
        
        self.df = self.df.sample(frac=1, random_state=0)
        
        if mode == "tiny":
            self.df = self.df.sample(frac=.05, random_state=0)
        
    
    def __getitem__(self, idx):
        file_name, label = self.df.iloc[idx]
        if not self.upsampled:
            img = np.load(file_name)[1:4]
            sr=cv2.dnn_superres.DnnSuperResImpl_create()
            path="../ESPCN_x4.pb"
            sr.readModel(path)
            sr.setModel('espcn',4)
            rgb=np.transpose(img)
            rgb=rgb-np.min(rgb)
            rgb=rgb/np.max(rgb)
            rgb = np.uint8(255 * rgb)
            result=sr.upsample(rgb)
            img = torch.Tensor(result/256).to(self.device)
        else:
            img = torch.Tensor(np.load(file_name)/256).to(self.device)
        
        #img = torch.Tensor(np.load(file_name)/256).to(self.device)
        if self.transform:
            img=self.transform(img)
            img=img*256
        else:
            img=img*256
        img.int()
            
        label = torch.Tensor([label]).to(self.device)
        
        return img, label
        
        
    def __len__(self):
        return len(self.df)
    

class Resnet(torch.nn.Module):
    def __init__(self, num_channels, image_width=64, device='cuda', pretrained=False):
        super(Resnet, self).__init__()
        self.device = device        
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.conv1 = torch.nn.Conv2d(num_channels, image_width, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 256).to(device)
        self.final_fc = torch.nn.Linear(256, 1).to(device)
        
        self.resnet = self.resnet.to(device)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        resnet_output = self.resnet(x)
        outputs = self.final_fc(resnet_output)
        outputs = self.sigmoid(outputs)
        return outputs
    
def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_iter, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(np.transpose(inputs.cpu(),(0,3,1,2)).to('cuda')).to('cuda')
        
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    
    return running_loss / (batch_iter+1)
        
        
def evaluate_model(model, val_loader,criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        all_labels = []
        all_preds = []
        all_outputs = []
        for batch_iter, (inputs, labels) in enumerate(val_loader):

            outputs = model(np.transpose(inputs.cpu(),(0,3,1,2)).to('cuda')).to('cuda')
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            preds = (outputs > .5).astype('int')
            
            all_labels.append(labels)
            all_preds.append(preds)
            all_outputs.append(outputs)
        
        all_labels = np.array([label for vec in all_labels for label in vec])
        all_preds = np.stack([pred for vec in all_preds for pred in vec])
        all_outputs = np.stack([output for vec in all_outputs for output in vec])
        val_acc = np.mean((all_labels == all_preds).astype('int'))
            
    return val_acc, running_loss / (batch_iter+1), all_labels, all_outputs


def save_checkpoint(logdir, model, optimizer, epoch, loss, lr, best=None,last=None):
    """
    Saves model checkpoint after each epoch
    logdir: Log directory
    model: model to save
    optimizer: optimizer to save
    epoch: epoch to save on
    loss: loss to log
    lr: learning rate to log
    best: An optional string used to specify which validation method this best
    checkpoint is for
    """
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if best:
        print(f"Saving checkpoint, {best}.")
        checkpoint_path = "{}/best_{}.pth".format(checkpoint_dir, best)
    elif last:
        print(f"Saving last, {last}.")
        checkpoint_path = "{}/last_{}.pth".format(checkpoint_dir,last)
    else:
        checkpoint_path = "{}/lr{}_epoch{}.pth".format(checkpoint_dir, lr, epoch)

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_path)

    if not best:
        print("Saving checkpoint to lr{}_epoch{}".format(lr, epoch))
        

def main(dataset_list,model_name,constants,transform_list=None):
    logdir=constants.MODEL_ROOT+model_name
    dataset=[]
    train_dset=None
    val_dset=None
    
    for dataset_name in dataset_list:
        dataset.append(Combo_BI_Dataset(dataset_name,constants,True,702,transform=transform_list))
        
    for dset in dataset:
        train, val = torch.utils.data.random_split(
            dset, 
            [len(dset)*8//10, len(dset)-len(dset)*8//10], # 80-20% split
            generator=torch.Generator().manual_seed(0)
        )
        if train_dset is None and val_dset is None:
            train_dset=train
            val_dset=val
        else:
            train_dset=torch.utils.data.ConcatDataset([train_dset,train])
            val_dset=torch.utils.data.ConcatDataset([val_dset,val])
    print(f"{len(train_dset)} training examples and {len(val_dset)} validation examples.")
    train_loader = DataLoader(train_dset, batch_size=64,shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=64,shuffle=True)
    
    model = Resnet(3, device='cuda')
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = None

    logdir = constants.MODEL_ROOT+model_name+"_50_training_steps/"
    
    early_stop=0
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    for epoch in tqdm(range(constants.TRAINING_N_EPOCHS)):
        if early_stop >= 10:
            print('Early stopping!')
            break;
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_loss, _, _ = evaluate_model(model, val_loader,criterion)

        print(f"Train_loss: {train_loss:.6f}, Val-acc: {val_acc:.5f}, Val-loss: {val_loss:.5f}")

        if epoch >= 5 and (best_val_loss is None or val_loss < best_val_loss):
            early_stop = 0
            best_val_loss = val_loss
            print("Best val loss, saving checkpoint!")
            save_checkpoint(logdir, model, optimizer, epoch, val_loss, lr, best="dl_best")
        elif epoch > 5 and (val_loss >= best_val_loss*1.1):
            early_stop+=1
        if epoch % 12 == 0:
            save_checkpoint(logdir, model, optimizer, epoch, val_loss, lr)
        print('Early Stop:',early_stop,'/10')
    print('Done!')
    save_checkpoint(logdir, model, optimizer, epoch, val_loss, lr,last='dl_last')

    
def verify_datasets(dataset_list,constants):
    fail_paths=[]
    for dataset in dataset_list:
        dataset_path=constants.IMAGE_DATASETS_ROOT+dataset
        if not os.path.exists(dataset_path):
            fail_paths.append(dataset_path)
    if fail_paths:
        print('One or more provided datasets does not exist, please update constants.py: '+str(fail_paths))
        return False
    return True

def verify_model_name(model_name,constants):
    if os.path.exists(constants.MODEL_ROOT+model_name+'_50_training_steps'):      
        print('Model already exists, delete the directory or choose a different name: '+constants.MODEL_ROOT+model_name)
    else:
        for char in model_name:
            if not char.isalnum() and char not in '-_ ':
                print('Provided model name contains a character that is not alphanumeric or one of "-_ ", please try again: '+model_name)
                return False
        return True

def initiate():
    args=sys.argv[1:]
    constants=None
    if any(x in ['-co', '-config'] for x in args):
        try:
            index=args.index('-co')
        except:
            pass
        try: 
            index=args.index('-config')
        except:
            pass
        file_path =str(pathlib.Path('../Configs/'+args[index+1]))
        module_name = args[index+1][:-3]
        del args[index:index+2]
    else:
        file_path =str(pathlib.Path('../Configs/constants.py'))
        module_name = 'constants'
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    if len(args)==1:
        if args[0] in ['-h','-help']:
            print(help_texts.TRAIN_TEXT)
            return
    elif len(args)%2==0:
        model=constants.TRAINING_MODEL_NAME
        dataset_list=constants.TRAINING_DATASET_LIST
        arg=[args[i:i + 2] for i in range(0, len(args), 2)]
        for instruction in arg:
            if instruction[0] in ['-model','-m']:
                model=instruction[1]
            else:
                print('Bad keyword, '+instruction[0]+', expected [-model,-m].')
                return
        if not verify_datasets(dataset_list,constants):
            return
        if not verify_model_name(model,constants):
            return
        main(dataset_list,model,constants)
    else:
        print('Expected an even number of arguments (up to 2) found: '+str(len(args)))

if __name__ == '__main__':
    initiate()
    
    