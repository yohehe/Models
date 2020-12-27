# Model Qautization
'''
Introduction to Quantization on PyTorch\
https://pytorch.org/blog/introduction-to-quantization-on-pytorch/
'''

#とりあえずモデルを作成する

#module
import numpy as np
import matplotlib.pyplot as plt

#pytorch
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os


#pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

#あやめ分類を利用する。
from sklearn.datasets import load_iris
iris_dataset=load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], 
    test_size=0.3,  random_state=0)
    
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_test)

class My_Dataset(Dataset):
    def __init__(self,data,target,transform=None):
        #dataframeを格納
        self.data = data
        self.target= target
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        #image_id
        #img_data=self.data[index][0]
        #torch.tensor
        #img_data=torch.tensor(img_data).clone().detach() / 255.0
    
        #flatten
        #img_data=img_data.flatten()
        #transform with albumentations
        #if self.transform is not None:
            #img_transformed=self.transform(image=img_rgb)["image"]
        
        inputs=self.data[index]
        inputs=np.array(inputs).astype(np.float32)
        
        #change channels first
        #img=np.einsum('ijk->kij', img_transformed)
        
        #label
        labels=self.target[index]
        #target
        
        return inputs,labels

#DataSet
train_dataset=My_Dataset(X_train,y_train,transform=None)
val_dataset=My_Dataset(X_test,y_test,transform=None)

#Dataloader
batch_size=10
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataloader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

#dict
dataloaders_dict={"train":train_dataloader,"val":val_dataloader}

#test_output
batch_iterator=iter(dataloaders_dict["train"])
inputs,labels=next(batch_iterator)
print(inputs.shape) 
print(labels.shape)
print(labels)


# Model
#settingをリスト化してモデルを作成可能に調整する
# 基本構造

class model_standard(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=10,out_dim=3,dropout=0.1):
        super().__init__()
        
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_1=nn.BatchNorm1d(hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        #linear->batchnorm->relu->dropout->relu->linear
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.linear_2(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

model=model_standard()
output=model(inputs)
output.shape
#>>>torch.Size([10, 3])

class EarlyStopping:
    def __init__(self,patience=0,verbose=0):
        self._step=0
        self._loss=float("inf")
        self.patience=patience
        self.verbose=verbose
    
    def __call__(self,loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print("early stopping")
                return True
            
        else:
            self._step=0
            self._loss=loss
        
        return False

#train
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # 初期設定
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True
    
    #early_stopping:
    es=EarlyStopping(patience=5,verbose=1)  #インスタンスを作成


#-----epochs-----
num_epochs=200
#-----models-----
net=model_standard()
#-----criterion-----
criterion=nn.CrossEntropyLoss()

#-----learning rate-----
learning_rate=0.0001

#-----optimizer-----
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

from tqdm import tqdm
#正答率、損失関すのグラフまで出力
train_model(net=net,
            dataloaders_dict=dataloaders_dict,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs)



    
    """
    とりあえず使用中止で
    #early_stopping:
    es=EarlyStopping(patience=5,verbose=1)  #インスタンスを作成
    """
    
    
    #dict形式で出力値をストック
    hist={"loss":[],"acc":[],"val_loss":[],"val_acc":[]}
    
    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            epoch_f1 = 0.
            epoch_recall=0.
            epoch_precision=0.

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # GPUが使えるならGPUにデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    outputs= net(inputs)
                    loss = criterion(outputs, labels)  # 損失を計算
                    
                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    
                    # 正解数の合計を更新
                    _,pred = torch.max(outputs.data, 1)
                    epoch_corrects += torch.sum(pred == labels.data)
                     
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            
            epoch_corrects=epoch_corrects.float()
            epoch_acc = epoch_corrects/ len(dataloaders_dict[phase].dataset)
            
            if phase=="train":
                hist["loss"].append(epoch_loss)
                hist["acc"].append(epoch_acc)
            else:
                hist["val_loss"].append(epoch_loss)
                hist["val_acc"].append(epoch_acc)
                
            print("{} Loss:{:.4f},Accuracy:{:.4f},".format(phase, epoch_loss, epoch_acc))
        
        #-----early_stopping:-----
        if phase=="val":
            if es(epoch_loss):
                print("early stopping")
                #グラフ表示の設定:
                fig,(axL,axR)=plt.subplots(ncols=2,figsize=(20,5))
                # plot learning curve
                plt.figure()
                #1回目の計算を取得しないようにしているため,-1を行う必要あり。
                axL.plot(hist["loss"],color='skyblue', label='loss')
                axL.plot(hist["val_loss"], color='orange', label='val_loss')
                axL.legend()
                axL.set_xlabel('epochs')
                axL.set_ylabel('loss')
                axL.grid(True)
                plt.figure()
                axR.plot(hist["acc"], color='skyblue', label='acc')
                axR.plot(hist["val_acc"], color='orange', label='val_acc')
                axR.legend()
                axR.set_xlabel('epochs')
                axR.set_ylabel('accuracy')
                axR.grid(True)
                plt.show()
                break
    else:
        print("No Use:EarlyStopping")
        #グラフ表示の設定:
        fig,(axL,axR)=plt.subplots(ncols=2,figsize=(20,5))
        # plot learning curve
        plt.figure()
        #1回目の計算を取得しないようにしているため,-1を行う必要あり。
        axL.plot(hist["loss"],color='skyblue', label='loss')
        axL.plot(hist["val_loss"], color='orange', label='val_loss')
        axL.legend()
        axL.set_xlabel('epochs')
        axL.set_ylabel('loss')
        axL.grid(True)
        plt.figure()
        axR.plot(hist["acc"], color='skyblue', label='acc')
        axR.plot(hist["val_acc"], color='orange', label='val_acc')
        axR.legend()
        axR.set_xlabel('epochs')
        axR.set_ylabel('accuracy')
        axR.grid(True)
        plt.show()

#ここからが量子化

#使い方がよくわからん。
import torch.quantization
output=net(torch.randn(10,4))
print(output)

#サイズ確認
#torchsummaryを用いてモデルサイズを調べる
import torchsummary
#torchsummary.summary(model,tuppleでinputs.shapeを入力する）
torchsummary.summary(net,(10,4))


