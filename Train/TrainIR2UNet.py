################### train AutoBCS framework #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import nibabel as nib
from TrainingDataLoad import *
from loss import * 
from IR2Unet import *
#########  Section 1: DataSet Load #############
def DataLoad(Batch_size):
    DATA_DIRECTORY = '//mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/CC/QSM_Recon/CC_code/codes_for_cc'
    DATA_LIST_PATH = '/mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/CC/QSM_Recon/CC_code/codes_for_cc/test_IDs.txt'
    
    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=True, drop_last = True)
    return trainloader

def TrainNet(Chi_Net, LR = 0.001, Batchsize = 32, Epoches = 100 , useGPU = True):
    print('DataLoader setting begins')
    trainloader = DataLoad(Batchsize)
    print('Dataloader settting end')

    print('Training Begins')
    criterion1 = nn.MSELoss(reduction='sum')
    criterion2 = Multi_Step_Loss(weight=0.5, n_inp=4, reduction='sum')
    optimizer1 = optim.Adam(Chi_Net.parameters())

    scheduler1 = LS.MultiStepLR(optimizer1, milestones = [40, 60], gamma = 0.1)
    ## start the timer. 
    time_start=time.time()
    best_loss = float('inf')
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0")

            Chi_Net = nn.DataParallel(Chi_Net)
            Chi_Net.to(device)

            for epoch in range(1, Epoches + 1):
                total_train_loss = 0.0

                if epoch % 20 == 0:
                    torch.save(Chi_Net.state_dict(), ("model_IR2Unet_%s.pth" % epoch))

                for i, data in enumerate(trainloader):
                    lfss, chis, name = data
                    lfss = lfss.to(device)
                    chis = chis.to(device)
                    ## zero the gradient buffers 
                    optimizer1.zero_grad()
                    ## forward: 
                    pred_chis, all_out = Chi_Net(lfss)
                    ## loss
                    loss1 = criterion1(pred_chis, chis)
                    loss2 = criterion2(all_out, chis)
                    loss = loss1+loss2
                    ## backward
                    loss.backward()
                    ##
                    optimizer1.step()
                    optimizer1.zero_grad()

                    total_train_loss += loss.item()

                    ## print every 20 mini-batch size
                    if i % 20 == 0:
                        acc_loss1 = loss.item()   
                        time_end=time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss: %f, lr1: %f,  used time: %d s' %
                            (epoch, i + 1, acc_loss1, optimizer1.param_groups[0]['lr'], time_end - time_start))  
                
                    if loss.item() < best_loss:
                        best_loss = loss
                        torch.save(Chi_Net.state_dict(), 'model_IR2unet_best.pth') 

                scheduler1.step()

                train_loss = total_train_loss / len(trainloader)

                torch.save(Chi_Net.state_dict(), 'model_IR2unet_latest.pth')
                with open('Epoch_Loss_IR2unet.txt','a') as file:
                    file.write(f"{epoch},{train_loss}\n")
        else:
            pass
            print('No Cuda Device!')
            quit()        
    print('Training Ends')


if __name__ == '__main__':
    Chi_Net = IR2Unet()
    Chi_Net.apply(weights_init)
    Chi_Net.train()
    print(get_parameter_number(Chi_Net)) 

    ## train network
    TrainNet(Chi_Net, LR = 0.001, Batchsize = 8, Epoches = 80 , useGPU = True)

