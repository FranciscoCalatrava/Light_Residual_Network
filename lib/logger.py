import wandb
from typing import List
from typeguard import typechecked 
import torch
import torch.nn as nn
#from torch.utils.tensorboard import summaryWrite


class Logger():
    @typechecked
    def __init__(self, type: str, logged_variable: List[str]) -> None:
        self.type = type #Define the type of logger that you want
        self.variables = logged_variable #It is a list with the variable that I want to track
        pass

    @typechecked
    def __getattribute__(self, __name: str):
        return super(Logger,self).__getattribute__(__name)



class WandB_logger(Logger):
    @typechecked
    def __init__(self, type: str, logged_variable: List[str], project: str, entity: str) -> None:
        super().__init__(type, logged_variable)
        self.project = project
        self.entity = entity
        self.client = wandb
        self.initialize()

    def initialize(self):
        ##*********************************************************************Inputs**********************************************************************##
        ## Description: Init the data logger
        ##.................................................................................................................................................##
        ## No inputs
        ##*********************************************************************Outputs*********************************************************************##
        ## No output
        ##*************************************************************************************************************************************************##
        self.client.init(project = self.project, entity = self.entity)

    @typechecked
    def hyperparameters(self, hyper: dict):
        ##*********************************************************************Inputs**********************************************************************##
        ## Description: Set the hyperparameter configuration of the training 
        ##.................................................................................................................................................##
        ## hyper --  It is a python dictionary with the 
        ##*********************************************************************Outputs*********************************************************************##
        ## No output
        ##*************************************************************************************************************************************************##
        self.client.config = hyper
    
    @typechecked
    def log(self, values: dict):
        ##*********************************************************************Inputs**********************************************************************##
        ## Description:  This function will log the data in the W&B
        ##.................................................................................................................................................##
        ## Values -- It is a python dictionary in which each key will be the same as the self.logged variable of the father class
        ##*********************************************************************Outputs*********************************************************************##
        ## No output
        ##*************************************************************************************************************************************************##
        self.client.log(values)
    
    def log_conf_matrix(self, pred, truth):
        m = nn.Softmax(dim=1)
        aux = torch.stack(pred,dim=0)
        new_dim = aux.size(0)*aux.size(1)
        m_1 = m(torch.reshape(aux,(new_dim,6)))
        # print(m_1)
        m_2 = torch.argmax(m_1, dim=1)
        pred_1 = m_2.detach().cpu().numpy()
        ################################################
        aux_1 = torch.stack(truth,dim=0)
        new_dim_1 = aux_1.size(0)*aux_1.size(1)
        m_3 = torch.argmax(torch.reshape(aux_1,(new_dim,6)), dim = 1)
        label_1 = m_3.detach().cpu().numpy()
        #####################################################################
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=label_1, preds=pred_1,
                            class_names=["0","1","2","3","4","5"])})

    
    def stop_log(self):
        ##*********************************************************************Inputs**********************************************************************##
        ## Description: Stop the data logger
        ##.................................................................................................................................................##
        ## No inputs
        ##*********************************************************************Outputs*********************************************************************##
        ## No output
        ##*************************************************************************************************************************************************##
        self.client.finish()