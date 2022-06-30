<<<<<<< HEAD
import numpy as np
import pandas as pd
import os
import cv2
import argparse
from platform import python_version
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize
import os
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import datetime
import wandb
'''

'''
#-------------------------------------------------------
# 0. Version Check
#-------------------------------------------------------
def verCheck():
    # 사용중인 라이브러리 버전 확인 : 라이브러리 버전이 맞지 않는 경우 코드가 실행되지 않을 수 있음
    print("[Version Check]")
    print("python version :", python_version())
    print("numpy version : ", np.__version__)
    print("pandas version : ", pd.__version__)
    # keras, sklearn은 전체를 import하지 않기 때문에 anaconda prompt에서 pip show scikit-learn, pip show keras로 확인
    print("sklearn version : ", "0.23.3")
    print("keras version : ", "2.4.3\n")
#-------------------------------------------------------
# 1. Load model
#-------------------------------------------------------
class MyNN(nn.Module):
    def __init__(self, modelname, inp_num):
        super(MyNN, self).__init__()
        self.modelname=modelname
        if modelname == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=False)
            # prev_w = self.model.conv1.weight
            self.model.conv1 = nn.Conv2d(inp_num, 64, kernel_size=7, padding=2, stride=3)
            # self.model.conv1.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64,1,7,7)), dim=1))
            self.classifier = nn.Linear(1000, output_num)
        elif modelname == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=False)
            # prev_w = self.model.conv1.weight
            self.model.conv1 = nn.Conv2d(inp_num, 64, kernel_size=7, padding=2, stride=3)
            # self.model.conv1.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64,1,7,7)), dim=1))
            self.classifier = nn.Linear(1000, output_num)
        elif modelname == 'resnet152':
            self.model = torchvision.models.resnet152(pretrained=False)
            # prev_w = self.model.conv1.weight
            self.model.conv1 = nn.Conv2d(inp_num, 64, kernel_size=7, padding=2, stride=3)
            # self.model.conv1.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64,1,7,7)), dim=1))
            self.classifier = nn.Linear(1000, output_num)
        elif modelname == 'resnet34':
            self.model = torchvision.models.resnet34(pretrained=False)
            self.model.conv1 = nn.Conv2d(inp_num, 64, kernel_size=7, padding=2, stride=3)
            self.classifier = nn.Linear(1000, output_num)
        elif modelname == 'initial':
            self.model = nn.Sequential(
                nn.Conv2d(inp_num, 32, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(3808064, 32),
                nn.ReLU())
            self.classifier = nn.Linear(32, output_num)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

def load_model(modelname, inp_num):
    model = MyNN(modelname, inp_num)
    print(f'{modelname} generated\n')

    model=model.to(device)
    return model
#-------------------------------------------------------
# 2. Set optimizer and loss function
#-------------------------------------------------------
def set_opt(model, weight_decay=0.):
    '''
        Set optimizer and loss fuction
    '''
    optimizer = Adam(params=model.parameters(),lr=lr,weight_decay=weight_decay)
    if output_num == 15:
        criterion = nn.MultiLabelSoftMarginLoss()
    elif output_num == 58:
        criterion = nn.CrossEntropyLoss()
    return optimizer,criterion
#-------------------------------------------------------
# 2. Load data
#-------------------------------------------------------
class ComponentDataset(Dataset):
    """Flow Field dataset."""

    def __init__(self, csv_file, root_dir, inp_type=['u','v','w','vor'], transform=None, mode=0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.affected_comp = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.u_exist = False
        self.v_exist = False
        self.w_exist = False
        self.vor_exist = False
        self.mode = mode
        for i in inp_type:
            if i == 'u':
                self.u_dir = root_dir + 'Velocity_U/'
                self.u_exist = True
            elif i == 'v':
                self.v_dir = root_dir + 'Velocity_V/'
                self.v_exist = True
            elif i == 'w':
                self.w_dir = root_dir + 'Velocity_W/'
                self.w_exist = True
            elif i == 'vor':
                self.vor_dir = root_dir + 'Vorticity_Mag/'
                self.vor_exist = True
        
        self.design_df_bin = pd.get_dummies(self.affected_comp['front overhang(%)'].astype(str) + self.affected_comp['stagnation width(%)'].astype(str) +\
                        self.affected_comp['front corner roundness'].astype(str) + self.affected_comp['side flat length(%)'].astype(str) +\
                        self.affected_comp['side flat angle'].astype(str) +\
                        self.affected_comp['front vertical angle'].astype(str) + self.affected_comp['height between stagnation to BLE(%)'].astype(str)+\
                        self.affected_comp['roof angle'].astype(str) +\
                        self.affected_comp['half roof angle'].astype(str) + self.affected_comp['end roof angle'].astype(str)+\
                        self.affected_comp['rr glass angle'].astype(str) + self.affected_comp['rr angle'].astype(str)+\
                        self.affected_comp['DLO boat tail angle'].astype(str) + self.affected_comp['DLO rr corner roundness'].astype(str)+\
                        self.affected_comp['defusing angle'].astype(str))

    def getNumTypes(self):
        return self.design_df_bin.shape[1]

    def __len__(self):
        return len(self.affected_comp)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 유동장 차이 이미지
        img_temp = np.zeros([1,402,602])
        if self.u_exist == True:
            u_img_name = self.affected_comp.iloc[idx, 0][5:]+'.png'
            u_img_dir = self.u_dir + u_img_name
            u_img = np.expand_dims(cv2.imread(u_img_dir, cv2.IMREAD_GRAYSCALE), axis=0)
            #u_img = np.expand_dims(cv2.resize(cv2.imread(u_img_dir, cv2.IMREAD_GRAYSCALE),(224, 224)), axis=0)
            img_temp = np.concatenate((img_temp, u_img), axis=0)
        if self.v_exist == True:
            v_img_name = self.affected_comp.iloc[idx, 0][5:]+'.png'
            v_img_dir = self.v_dir + v_img_name
            v_img = np.expand_dims(cv2.imread(v_img_dir, cv2.IMREAD_GRAYSCALE), axis=0)
            #v_img = np.expand_dims(cv2.resize(cv2.imread(v_img_dir, cv2.IMREAD_GRAYSCALE),(224, 224)), axis=0)
            img_temp = np.concatenate((img_temp, v_img), axis=0)
        if self.w_exist == True:
            w_img_name = self.affected_comp.iloc[idx, 0][5:]+'.png'
            w_img_dir = self.w_dir + w_img_name
            w_img = np.expand_dims(cv2.imread(w_img_dir, cv2.IMREAD_GRAYSCALE), axis=0)
            #w_img = np.expand_dims(cv2.resize(cv2.imread(w_img_dir, cv2.IMREAD_GRAYSCALE),(224, 224)), axis=0)
            img_temp = np.concatenate((img_temp, w_img), axis=0)
        if self.vor_exist == True:
            vor_img_name = self.affected_comp.iloc[idx, 0][5:]+'.png'
            vor_img_dir = self.vor_dir + vor_img_name
            vor_img = np.expand_dims(cv2.imread(vor_img_dir, cv2.IMREAD_GRAYSCALE), axis=0)
            #vor_img = np.expand_dims(cv2.resize(cv2.imread(vor_img_dir, cv2.IMREAD_GRAYSCALE),(224, 224)), axis=0)
            img_temp = np.concatenate((img_temp, vor_img), axis=0)
        image = img_temp[1:,:,:]

        # 디자인 차이 데이터
        # 모든 경우의 수
        # dataLoader0을 사용
        if self.mode == 0:
            components = self.affected_comp.iloc[idx, 2:]
            components = np.array(components,dtype='float32')

        # 현존하는 경우의 수만 판단
        # 이 경우 dataLoader1을 사용
        elif self.mode == 1:
            components = self.design_df_bin.iloc[idx, :]
            components = np.array(components, dtype='float32')


        if self.transform:
            image = self.transform(image)

        return torch.from_numpy(image).type(torch.FloatTensor), torch.from_numpy(components)


def dataLoader0(train_path, test_path, input_type=['u','v','w','vor']):
    '''
        Load train, validation dataset
        A ratio of train set and val set is 9:1.
        이미 Train과 Testset으로 나눠둔 경우에 사용
    '''
    dataset= ComponentDataset(csv_file = train_path + 'design_diff_train.csv', root_dir = train_path, inp_type=input_type)
    test_set = ComponentDataset(csv_file = test_path + 'design_diff_test.csv', root_dir = test_path, inp_type=input_type)
    train_loader=DataLoader(dataset,batch_size=batchsize,shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    print('[Length of Loader]')
    print(' Train : {}\n Test : {}'.format(len(train_loader.dataset), len(test_loader.dataset)))

    return train_loader, test_loader

def dataLoader1(path, input_type):
    '''
        Load train, validation dataset
        A ratio of train set and val set is 9:1.
        Trainset과 Testset으로 나누지 않은 경우에 사용
    '''
    dataset = ComponentDataset(csv_file=path+'design_diff.csv', root_dir=path, inp_type=input_type)
    numTypes = dataset.getNumTypes()

    train_num = int(len(dataset) * 0.9)
    test_num = len(dataset) - train_num
    trainset, testset = torch.utils.data.random_split(dataset, [train_num, test_num])
    train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True)

    print('[Length of Loader]')
    print(' Train : {}\n Test : {}'.format(len(train_loader.dataset), len(test_loader.dataset)))

    return train_loader, test_loader
#-------------------------------------------------------
# 3. Train
#-------------------------------------------------------
def train_model(model, train_loader,criterion,optimizer,epoch,dataclass):
    '''
        Train model
    '''
    model.train()
    minibatches=len(train_loader)
    train_loss=[]

    for idx,(image,label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output=model(image)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        if idx%minibatches == 0:
            train_loss.append(loss.item())
            print("\nTrain Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(epoch+1,
            epoch, epochs, 100.*epoch / epochs,
            loss.item()))
            
    return train_loss

def evaluate_model(model,val_loader,criterion,model_path,dataclass):
    '''
        Evaluate model performance by 
        returning validation/test accuracy using validation/test set
    '''
    torch.save(model.state_dict(), model_path)
    model.eval()
    val_loss=0
    correct=0
    val_correct=0
    with torch.no_grad():
        for image,label in val_loader:
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            val_loss += criterion(output,label).item()
            output = output > 0

            if output_num == 15:
                if torch.eq(output, label).all() == True:
                    correct += 1
            elif output_num == 58:
                _, pred = output.max(1,keepdim=True)
                _, olabel = label.max(1,keepdim=True)
                correct += (pred == olabel.to(device).data.expand_as(pred)).float().sum().item()
            
            val_correct += (output == label).float().mean()

    val_acc = 100. * val_correct / len(val_loader)
    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / len(val_loader)
    return val_loss, val_accuracy, val_acc


def save_loss_plot(loss1,loss2,result_path):
    '''
        Save an image of Train, Validation Loss plots
    '''
    d=datetime.datetime.now()
    x=np.arange(1,epochs+1)
    fig,axes = plt.subplots(2,1)

    # Draw plot
    axes[0].plot(x,loss1)
    axes[1].plot(x,loss2)

    # X,Y labeling
    fig.text(0.5, 0.015, 'Epoch', ha='center', va='center')
    fig.text(0.02, 0.5, 'Loss', ha='center', va='center', rotation='vertical')

    # Set title
    axes[0].set_title('Train Loss')
    axes[1].set_title('Val Loss')

    plt.tight_layout()
    #fig.savefig(result_path+'Train1_loss_{}_{}_{}_{}.png'.format(epochs,resize,lr,d.strftime('%y%m%d')),dpi=300)
    fig.savefig(result_path+'Train1_loss_{}_{}_{}.png'.format(epochs,lr,d.strftime('%y%m%d')),dpi=300)

def train(args):
    d = datetime.datetime.now()

    # global variables for device, epochs, learning rate, resize, batchsize
    global device, epochs, lr, resize, batchsize, dataclass, output_num
    #global device, epochs, lr, batchsize, dataclass
    path = args.filepath
    lr = args.lr
    epochs = args.epoch
    batchsize=args.batchsize
    input_type = args.input
    mode = args.opt
    if mode==0:
        output_num = 15
    elif mode==1:
        output_num = 58
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if '*' in args.size:
            resize = tuple(map(int,args.size.split('*')))
        else:
            resize = int(args.size)
    except:
        "Wrong input size error"
        
    modelname = args.model
    dataclass = 15
    print('Device: {}\n'.format(device))

    result_path = path+'Results/'
    # Saved model name is model_epochs_learning rate_tuning layer size_datetime
    # model_path = path+'Results/model_{}_{}_{}_3_{}.pt'.format(epochs,resize,lr,d.strftime('%y%m%d'))
    model_path = path+'Results/{}_{}_{}_{}.pt'.format(modelname,epochs,lr,d.strftime('%y%m%d%H'))
    os.makedirs(result_path,exist_ok=True)

    # Load model
    model = load_model(modelname, len(input_type))

    # Dataload
    if mode == 0:
        train_loader,val_loader = dataLoader0(path+'Trainset/', path+'Testset/', input_type=input_type)
    elif mode == 1:
        train_loader,val_loader = dataLoader1(path, input_type=input_type)

    # Set optimizer and Loss function
    optimizer,criterion=set_opt(model)

    # wandb
    wandb.watch(model)

    train_loss_lst=[]
    val_loss_lst=[]
    val_acc_lst=[]
    # Scheduler multiply gamma to learning rate in every step_size
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Train
    for epoch in tqdm(range(epochs)):
        train_loss=train_model(model,train_loader,criterion,optimizer,epoch,dataclass)
        val_loss,val_acc, val_acc2=evaluate_model(model,val_loader,criterion,model_path,dataclass)
        print('\n[Epoch:{}], \tVal Loss: {:.4f},\tVal Acc: {:.3f} \tVal Acc2: {:.3f}% \n'.format(epoch+1,val_loss,val_acc, val_acc2))
        train_loss_lst.append(sum(train_loss)/len(train_loss))
        val_loss_lst.append(val_loss)
        val_acc_lst.append(val_acc)
        scheduler.step()
        wandb.log({'Epoch': epoch, 'Val_Acc': val_acc, 'Val_Loss': val_loss, 'Val_Acc2': val_acc2})

    save_loss_plot(train_loss_lst,val_loss_lst,result_path)

if __name__ =='__main__':
    verCheck()
    # WanDB로 데이터 저장
    # 직접 돌리실 때는 entity와 project 바꿔서 돌리시거나, wandb 부분 코드를 전부 제거하고 돌리셔도 됩니다.
    wandb.init(project='hyundai', entity='yunch')

    parser = argparse.ArgumentParser(description='detecting differences in design factors from flow fields')
    parser.add_argument('-b','--batchsize',type=int,default=32,help='default batch size is 1')
    parser.add_argument('-e', '--epoch', type=int, default='100',help='default 30')
    parser.add_argument('-l', '--lr', type=float, default='1e-5',help='defalt value 1e-4')
    parser.add_argument('-f', '--filepath', type=str, default='dataset/')
    # 실행 시 input의 default=['u'], ['v'], ['w'], ['vor'], ['u','v','w','vor] 중에 골라서 넣으면 됨
    parser.add_argument('-i', '--input', type=list, default=['u','v','w','vor'], help='choose from u,v,w,vor, default vor')
    parser.add_argument('-s', '--size', type=str, default='402*602',help='default 402 * 602')
    parser.add_argument('-m', '--model', type=str, default='resnet50',help='choose from resnet18, resnet50, resnet152')
    parser.add_argument('-o', '--opt', type=int, default=0, help='classfication mode 0 or 1, default=0')
    args = parser.parse_args()
    
    wandb.run.name = datetime.datetime.now().strftime('%y%m%d_%H%M') +'_'+ args.model +'_'+ str(args.lr)
    wandb.config.update(args)
=======
import numpy as np
import pandas as pd
import os
import cv2
import argparse
from platform import python_version
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize
import os
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import datetime
import wandb
'''

'''
#-------------------------------------------------------
# 0. Version Check
#-------------------------------------------------------
def verCheck():
    # 사용중인 라이브러리 버전 확인 : 라이브러리 버전이 맞지 않는 경우 코드가 실행되지 않을 수 있음
    print("[Version Check]")
    print("python version :", python_version())
    print("numpy version : ", np.__version__)
    print("pandas version : ", pd.__version__)
    # keras, sklearn은 전체를 import하지 않기 때문에 anaconda prompt에서 pip show scikit-learn, pip show keras로 확인
    print("sklearn version : ", "0.23.3")
    print("keras version : ", "2.4.3\n")
#-------------------------------------------------------
# 1. Load model
#-------------------------------------------------------
class MyNN(nn.Module):
    def __init__(self, modelname, inp_num):
        super(MyNN, self).__init__()
        if modelname == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=True)
            prev_w = self.model.conv1.weight
            self.model.conv1 = nn.Conv2d(inp_num, 64, kernel_size=7, padding=2, stride=3)
            self.model.conv1.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64,1,7,7)), dim=1))
            self.classifier = nn.Linear(1000, 15)
        elif modelname == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True)
            prev_w = self.model.conv1.weight
            self.model.conv1 = nn.Conv2d(inp_num, 64, kernel_size=7, padding=2, stride=3)
            self.model.conv1.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64,1,7,7)), dim=1))
            self.classifier = nn.Linear(1000, 15)
        elif modelname == 'resnet152':
            self.model = torchvision.models.resnet152(pretrained=True)
            prev_w = self.model.conv1.weight
            self.model.conv1 = nn.Conv2d(inp_num, 64, kernel_size=7, padding=2, stride=3)
            self.model.conv1.weight = nn.Parameter(torch.cat((prev_w, torch.zeros(64,1,7,7)), dim=1))
            self.classifier = nn.Linear(1000, 15)
        elif modelname == 'initial':
            self.model = nn.Sequential(
                nn.Conv2d(inp_num, 32, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(3808064, 32),
                nn.ReLU(),
                nn.Linear(32, 15))
    def forward(self, x):
        x = self.model(x)
        return x

def load_model(modelname, inp_num):
    model = MyNN(modelname, inp_num)
    print(f'{modelname} generated')

    model=model.to(device)
    return model
#-------------------------------------------------------
# 2. Set optimizer and loss function
#-------------------------------------------------------
def set_opt(model, weight_decay=0.):
    '''
        Set optimizer and loss fuction
    '''
    optimizer = Adam(params=model.parameters(),lr=lr,weight_decay=weight_decay)
    criterion = nn.MultiLabelSoftMarginLoss()
    return optimizer,criterion
#-------------------------------------------------------
# 2. Load data
#-------------------------------------------------------
class ComponentDataset(Dataset):
    """Flow Field dataset."""

    def __init__(self, csv_file, root_dir, inp_type=['u','v','w','vol'], transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.affected_comp = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.u_exist = False
        self.v_exist = False
        self.w_exist = False
        self.vor_exist = False
        for i in inp_type:
            if i == 'u':
                self.u_dir = root_dir + 'Velocity_U/'
                self.u_exist = True
            elif i == 'v':
                self.v_dir = root_dir + 'Velocity_V/'
                self.v_exist = True
            elif i == 'w':
                self.w_dir = root_dir + 'Velocity_W/'
                self.w_exist = True
            elif i == 'vol':
                self.vor_dir = root_dir + 'Vorticity_Mag/'
                self.vor_exist = True
            else: raise

    def __len__(self):
        return len(self.affected_comp)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_temp = np.zeros([1,402,602])
        if self.u_exist == True:
            u_img_name = self.affected_comp.iloc[idx, 0][5:]+'.png'
            u_img_dir = self.u_dir + u_img_name
            u_img = np.expand_dims(cv2.imread(u_img_dir, cv2.IMREAD_GRAYSCALE), axis=0)
            #u_img = np.expand_dims(cv2.resize(cv2.imread(u_img_dir, cv2.IMREAD_GRAYSCALE),(224, 224)), axis=0)
            img_temp = np.concatenate((img_temp, u_img), axis=0)
        if self.v_exist == True:
            v_img_name = self.affected_comp.iloc[idx, 0][5:]+'.png'
            v_img_dir = self.v_dir + v_img_name
            v_img = np.expand_dims(cv2.imread(v_img_dir, cv2.IMREAD_GRAYSCALE), axis=0)
            #v_img = np.expand_dims(cv2.resize(cv2.imread(v_img_dir, cv2.IMREAD_GRAYSCALE),(224, 224)), axis=0)
            img_temp = np.concatenate((img_temp, v_img), axis=0)
        if self.w_exist == True:
            w_img_name = self.affected_comp.iloc[idx, 0][5:]+'.png'
            w_img_dir = self.w_dir + w_img_name
            w_img = np.expand_dims(cv2.imread(w_img_dir, cv2.IMREAD_GRAYSCALE), axis=0)
            #w_img = np.expand_dims(cv2.resize(cv2.imread(w_img_dir, cv2.IMREAD_GRAYSCALE),(224, 224)), axis=0)
            img_temp = np.concatenate((img_temp, w_img), axis=0)
        if self.vor_exist == True:
            vor_img_name = self.affected_comp.iloc[idx, 0][5:]+'.png'
            vor_img_dir = self.vor_dir + vor_img_name
            vor_img = np.expand_dims(cv2.imread(vor_img_dir, cv2.IMREAD_GRAYSCALE), axis=0)
            #vor_img = np.expand_dims(cv2.resize(cv2.imread(vor_img_dir, cv2.IMREAD_GRAYSCALE),(224, 224)), axis=0)
            img_temp = np.concatenate((img_temp, vor_img), axis=0)
        image = img_temp[1:,:,:]
        components = self.affected_comp.iloc[idx, 2:]
        components = np.array(components,dtype='float32')
        
        if self.transform:
            image = self.transform(image)

        return torch.from_numpy(image).type(torch.FloatTensor), torch.from_numpy(components)


def dataLoader(train_path, test_path):
    '''
        Load train, validation dataset
        A ratio of train set and val set is 9:1. 
    '''
    dataset= ComponentDataset(csv_file = train_path + 'design_diff_train.csv', root_dir = train_path)
    test_set = ComponentDataset(csv_file = test_path + 'design_diff_test.csv', root_dir = test_path)
    # len_train=int(len(dataset)*0.9)
    # len_val=len(dataset)-len_train
    # train_set, val_set = torch.utils.data.random_split(dataset, [len_train,len_val])
    # train_loader=DataLoader(train_set,batch_size=batchsize,shuffle=True)
    train_loader=DataLoader(dataset,batch_size=batchsize,shuffle=True)
    # val_loader=DataLoader(val_set,batch_size=batchsize,shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=True)
    
    print('[Length of Loader]')
    # print(' Train : {}\n Val : {}'.format(len(train_loader.dataset), len(val_loader.dataset)))
    print(' Train : {}\n Test : {}'.format(len(train_loader.dataset), len(test_loader.dataset)))

    return train_loader, test_loader
#-------------------------------------------------------
# 3. Train
#-------------------------------------------------------
def train_model(model, train_loader,criterion,optimizer,epoch,dataclass):
    '''
        Train model
    '''
    model.train()
    minibatches=len(train_loader)
    train_loss=[]

    for idx,(image,label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output=model(image)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        if idx%minibatches == 0:
            train_loss.append(loss.item())
            print("\nTrain Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(epoch+1,
            epoch, epochs, 100.*epoch / epochs,
            loss.item()))
            
    return train_loss

def evaluate_model(model,val_loader,criterion,model_path,dataclass):
    '''
        Evaluate model performance by 
        returning validation/test accuracy using validation/test set
    '''
    torch.save(model.state_dict(), model_path)
    model.eval()
    val_loss=0
    correct=0

    with torch.no_grad():
        for image,label in val_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            val_loss += criterion(output,label).item()

            output = output > 0
            #result = result.to(device)
            correct+= int(torch.eq(output,label).all())

    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / len(val_loader)
    return val_loss, val_accuracy


def save_loss_plot(loss1,loss2,result_path):
    '''
        Save an image of Train, Validation Loss plots
    '''
    d=datetime.datetime.now()
    x=np.arange(1,epochs+1)
    fig,axes = plt.subplots(2,1)

    # Draw plot
    axes[0].plot(x,loss1)
    axes[1].plot(x,loss2)

    # X,Y labeling
    fig.text(0.5, 0.015, 'Epoch', ha='center', va='center')
    fig.text(0.02, 0.5, 'Loss', ha='center', va='center', rotation='vertical')

    # Set title
    axes[0].set_title('Train Loss')
    axes[1].set_title('Val Loss')

    plt.tight_layout()
    #fig.savefig(result_path+'Train1_loss_{}_{}_{}_{}.png'.format(epochs,resize,lr,d.strftime('%y%m%d')),dpi=300)
    fig.savefig(result_path+'Train1_loss_{}_{}_{}.png'.format(epochs,lr,d.strftime('%y%m%d')),dpi=300)

def train(args):
    d = datetime.datetime.now()

    # global variables for device, epochs, learning rate, resize, batchsize
    global device, epochs, lr, resize, batchsize, dataclass
    #global device, epochs, lr, batchsize, dataclass
    path = args.filepath
    lr = args.lr
    epochs = args.epoch
    batchsize=args.batchsize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if '*' in args.size:
            resize = tuple(map(int,args.size.split('*')))
        else:
            resize = int(args.size)
    except:
        "Wrong input size error"
        
    modelname = args.model
    dataclass = 15
    print(device)

    result_path = path+'Results/'
    # Saved model name is model_epochs_learning rate_tuning layer size_datetime
    # model_path = path+'Results/model_{}_{}_{}_3_{}.pt'.format(epochs,resize,lr,d.strftime('%y%m%d'))
    model_path = path+'Results/model_{}_{}_3_{}.pt'.format(epochs,lr,d.strftime('%y%m%d'))
    os.makedirs(result_path,exist_ok=True)

    # Load model
    model = load_model(modelname, 4)

    # Dataload
    train_loader,val_loader = dataLoader(path+'Trainset/', path+'Testset/')

    # Set optimizer and Loss function
    optimizer,criterion=set_opt(model)

    # wandb
    wandb.watch(model)

    train_loss_lst=[]
    val_loss_lst=[]
    val_acc_lst=[]
    # Scheduler multiply gamma to learning rate in every step_size
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Train
    for epoch in tqdm(range(epochs)):
        train_loss=train_model(model,train_loader,criterion,optimizer,epoch,dataclass)
        val_loss,val_acc=evaluate_model(model,val_loader,criterion,model_path,dataclass)
        print('\n[Epoch:{}], \tVal Loss: {:.4f},\tVal Acc: {:.3f} % \n'.format(epoch+1,val_loss,val_acc))
        train_loss_lst.append(sum(train_loss)/len(train_loss))
        val_loss_lst.append(val_loss)
        val_acc_lst.append(val_acc)
        scheduler.step()
        wandb.log({'Epoch': epoch+1, 'Val_Acc': val_acc, 'Val_Loss': val_loss})

    save_loss_plot(train_loss_lst,val_loss_lst,result_path)

if __name__ =='__main__':
    verCheck()
    wandb.init(project='hyundai', entity='yunch')

    parser = argparse.ArgumentParser(description='detecting differences in design factors from flow fields')
    parser.add_argument('-b','--batchsize',type=int,default=1,help='default batch size is 1')
    parser.add_argument('-e', '--epoch', type=int, default='100',help='default 100')
    parser.add_argument('-l', '--lr', type=float, default='1e-4',help='defalt value 1e-4')
    parser.add_argument('-f', '--filepath', type=str, default='dataset/')
    parser.add_argument('-s', '--size', type=str, default='224*224',help='120, 200, 120*120, must be larger than 112')
    parser.add_argument('-m', '--model', type=str, default='initial',help='choose from resnet18, resnet50, resnet152')
    args = parser.parse_args()
    
    wandb.run.name = datetime.datetime.now().strftime('%y%m%d_%H%M') +'_'+ args.model +'_'+ str(args.lr)
    wandb.config.update(args)
>>>>>>> efb2b4ac1f288df262e55a9946ec71f22c0715cf
    train(args)