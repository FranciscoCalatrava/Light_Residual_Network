import numpy as np
import torch
from lib.dataset import load_dataset
import torch.nn as nn
from torch.optim import Adam
from lib.normalization import normalization_data
import sys
from lib.logger import WandB_logger
from tqdm import trange, tqdm
from time import sleep
from sklearn.metrics import roc_auc_score
import datetime
import os
from sklearn.metrics import f1_score
import wandb
import os


class ResNetBranch_1D(nn.Module):
    def __init__(self, input_shape, num_blocks):
        super(ResNetBranch_1D, self).__init__()
        in_channels, height, width = input_shape

        self.in_channels = in_channels
        self.height = height
        self.width = width

        self.conv1 = nn.Conv2d(9, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0,2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size = (1,3), stride=(1,2), padding=(0,1), dilation=1, return_indices=False, ceil_mode=False)

        self.layer1 = self._make_layer_1D(64, num_blocks[0],  stride=1)
        self.layer2 = self._make_layer_1D(128, num_blocks[1], stride=(1,2))


        self.adpavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128,6)

    def _make_layer_1D(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = nn.ModuleList()
        layers.append(BasicBlock_1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock_1D.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock_1D(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.adpavgpool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return x

class BasicBlock_1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock_1D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=stride, padding=(0,1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU()
        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


#########################################################################################################################################################
def accurancy_1_1(output, label):
    ##*********************************************************************Inputs**********************************************************************##
    ## Description:  This function return the accuracy from the input data
    ##.................................................................................................................................................##
    ## output -- It is a tensor with the predictions from our model. The shape of this tensor will be ()                                               ##
    ## label --  It is the labels of our data. The shape will be ()                                                                                    ##
    ##*********************************************************************Outputs*********************************************************************##
    ## Accurancy -- It is the acc calculated by hand. It will be a tensor of 1x1 dimension                                                             ##
    ## aux_acc --  It is the acc calculated using torchmetrics                                                                                         ##
    ##*************************************************************************************************************************************************##
    m = nn.Softmax(dim=1)
    m_1 = m(output)
    m_2 = torch.argmax(m_1, dim=1)
    corrects = (m_2 == label)
    accurancy = corrects.sum().float()/float(label.shape[0])
    return accurancy
##########################################################################################################################################################



##########################################################################################################################################################
def saveModel(model):
    ##*********************************************************************Inputs**********************************************************************##
    ## Description: It is save the model in an specific path:  yearmonthday-hourminutessencods                                                         ##
    ##.................................................................................................................................................##
    ## model -- The model that i am training                                                                                                           ##
    ##*********************************************************************Outputs*********************************************************************##
    ## Path -- The path in which I have save the model                                                                                                 ##
    ##*************************************************************************************************************************************************##
    path = "./models/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"pid_"+str(os.getpid())
    torch.save(model, path)
    return path 
##########################################################################################################################################################


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(scheduler,learning_rate, num_epochs, batch_size, model, loss_fn, optimizer, X_train, labels_train):
    wand_log = WandB_logger(type= "wandb", logged_variable=["acc","loss"],project="xxxx", entity="xxxxx")
    wandb_config = {
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "batch_size": batch_size
    }
    wand_log. hyperparameters(wandb_config)

    # WE have to the define the device in wich we are going to run the training. It could be GPU or CPU dependig of what is available
    if torch.cuda.is_available():
        model.cuda()
    loss_ret = []   
    # Convert model parameters and buffers to CPU or Cuda
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        pred = []
        out = []
        delim = int(labels_train.shape[0]/batch_size)
        batch_start = 0
        cont = 1
        indices = np.arange(X_train.shape[0])
        np.random.seed(epoch)
        np.random.shuffle(indices)
        print(indices)
        X = X_train[indices,:,:,:].clone().detach()
        label_1 = labels_train[indices,:].clone().detach()
        with trange(delim, unit = "batch") as tepoch:
            for i in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                if (batch_start+batch_size <= labels_train.shape[0]):
                    images_1 = X[batch_start:batch_start+batch_size,:,:,:].cuda()
                    labels_1 = torch.squeeze(label_1[batch_start:batch_start+batch_size,:]).cuda()
                else:
                    images_1 = X[batch_start:,:,:,:,:].cuda()
                    labels_1 = torch.squeeze(label_1[batch_start:,:]).cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                # predict classes using images from the training set
                outputs = model(images_1)
                accurancy_1 = accurancy_1_1(outputs,labels_1)
                # compute the loss based on model output and real labels
                loss = loss_fn(outputs, labels_1)
                # backpropagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                optimizer.step()
                # Let's print statistics for every 1,000 images
                running_loss += loss.item()    # extract the loss value
                loss_ret.append(running_loss)
                batch_start = batch_start+batch_size
                running_acc+=accurancy_1             
                #pl.LightningModule.log('train_acc_step', aux_acc)
                tepoch.set_postfix(loss=loss.item(), accurancy_avg = running_acc/cont, lr = scheduler.get_lr())
                wand_log.log({"Train_LB": loss.item(),
                              "Train_LBM": loss.item()/cont,
                              "Train_AccB": accurancy_1,
                              "Train_AccBM": running_acc/cont
                              })
                cont+=1
                pred.append(outputs)
                out.append(labels_1)
                sleep(0.1)
        wand_log.log({"Train_LE": running_loss/delim,
                      "Train_RAccE": running_acc,
                      "Train_AccE": running_acc/delim})
        scheduler.step()
    path = saveModel(model=model)
    wand_log.stop_log()
    return loss_ret,path

#########################################################################################################################################################
def accurancy(output, label):
    ##*********************************************************************Inputs**********************************************************************##
    ## Description:  This function return the accuracy from the input data
    ##.................................................................................................................................................##
    ## output -- It is a tensor with the predictions from our model. The shape of this tensor will be ()                                               ##
    ## label --  It is the labels of our data. The shape will be ()                                                                                    ##
    ##*********************************************************************Outputs*********************************************************************##
    ## Accurancy -- It is the acc calculated by hand. It will be a tensor of 1x1 dimension                                                             ##
    ## aux_acc --  It is the acc calculated using torchmetrics                                                                                         ##
    ##*************************************************************************************************************************************************##
    m = nn.Softmax(dim=1)
    m_1 = m(output)
    m_2 = torch.argmax(m_1, dim=1)
    corrects = (m_2 == label)
    accurancy = corrects.sum().float() 
    return accurancy
##########################################################################################################################################################


def F1_score_1(pred,label, average):
    ############################################
    m = nn.Softmax(dim=1)
    aux = torch.squeeze(torch.stack(pred,dim=0))
    m_2 = m(aux)
    m_2_2 = torch.argmax(m_2, dim=1)
    pred_1 = m_2_2.detach().cpu().numpy()
    ################################################
    aux_1 = torch.stack(label,dim=0)
    label_1 = aux_1.detach().cpu().numpy()
    #####################################################################
    return np.squeeze(pred_1), np.squeeze(label_1), f1_score(np.squeeze(pred_1),np.squeeze(label_1), average= average)

# Function to test the model 
def test(path,batch_size,X_test, y_test):
    wand_log = WandB_logger(type= "wandb", logged_variable=["acc","loss"],project="smc-paper", entity="aiforlife")
    path = path
    model = torch.load(path)
     
    running_accuracy = 0 
    total = 0 
    if torch.cuda.is_available():
        model.cuda()
    with torch.no_grad():
        for epoch in range(1):
            model.eval()
            batch_start = 0
            cont =1
            out = []
            pred = []
            with trange(y_test.shape[0], unit = "sample") as tepoch:
                for i in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    if torch.cuda.is_available():
                        inputs = X_test[batch_start:batch_start+batch_size,:,:,:].cuda()
                        outputs = y_test[batch_start,:].cuda()
                    target = model(inputs)
                    accurancy_1 = accurancy(output=target,label=outputs)
                    running_accuracy+=accurancy_1
                    batch_start = batch_start+batch_size
                    tepoch.set_postfix(accurancy = accurancy_1)
                    wand_log.log({"Test_AccB": accurancy_1,
                              "Test_AccBM": running_accuracy/cont,
                              })
                    sleep(0.1)
                    total+=1
                    cont +=1
                    pred.append(target)
                    out.append(outputs)
    pred_1, label_1, f1_1 = F1_score_1(pred, out, average='micro')
    wand_log.log({"f1_micro": f1_1})
    wand_log.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
    y_true=label_1, preds=pred_1,
    class_names=["0","1","2","3","4","5"])})
    print("path...", path)
    wand_log.stop_log()     
 
# from lib.read_dataset_roman import DataReader, load_dataset
def reshape_dataset(trainX, testX):
    new_trainX = np.transpose(trainX,(0,2,1))
    new_trainX = new_trainX[:, :, np.newaxis,:]
    new_testX = np.transpose(testX,(0,2,1))
    new_testX = new_testX[:,:,np.newaxis,:]
    return new_trainX, new_testX


class Experiments():
    def __init__(self, experiment: int, hyperparameters: dict, shuffle: bool, channels: tuple, loo: float) -> None:
        self.experiment = experiment
        self.hyperparamenters = hyperparameters
        self.trainX = np.load('./Data_LOO/LOO_'+str(int(loo))+'_trainX.npy')
        self.trainy = np.load('./Data_LOO/LOO_'+str(int(loo))+'_trainy.npy')
        self.testX = np.load('./Data_LOO/LOO_'+str(int(loo))+'_testX.npy')
        self.testy = np.load('./Data_LOO/LOO_'+str(int(loo))+'_testy.npy')
        self.trainX_norm, self.testX_norm = normalization_data(trainX=self.trainX, testX=self.testX)
        self.valX_norm = None
        self.valy = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.shuffle = shuffle
        self.channels = channels
        self.sigma = loo
        pass
    def experiment_0(self):
        self.model = ResNetBranch_1D((64,1,128),[2,2,0,0])
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.hyperparamenters["lr"], weight_decay=0.0001)
        train_X, test_X = reshape_dataset(self.trainX_norm, self.testX_norm)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.4)
        loss_ret,path = train(scheduler ,self.hyperparamenters["lr"],self.hyperparamenters["epoch"],self.hyperparamenters["batch"],self.model,self.loss_fn,self.optimizer, torch.from_numpy(train_X[:,self.channels[0]:self.channels[1],:,:]).float(), torch.from_numpy(np.reshape(self.trainy,(self.trainy.shape[0],1))))
        test(path=path, batch_size = 1, X_test = torch.from_numpy(test_X[:,self.channels[0]:self.channels[1],:,:]).float(), y_test = torch.from_numpy(np.reshape(self.testy,(self.testy.shape[0],1))))

if __name__ == "__main__":
    experiments = int(sys.argv[1])
    hyperparameters = {"lr":float(sys.argv[2]), "epoch": int(sys.argv[3]), "batch":int(sys.argv[4])}
    channels_start = int(sys.argv[5])
    channels_finish = int(sys.argv[6])
    channels =(channels_start, channels_finish)
    loo = float(sys.argv[7])
    Experiments_instance = Experiments(experiment= experiments,hyperparameters= hyperparameters , shuffle = False, channels = channels, loo = loo)
    Experiments_instance.experiment_0()
