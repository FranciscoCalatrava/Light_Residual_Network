import numpy as np



def normalization_data(trainX, testX):
    ##***********************************************Inputs****************************************************##
    ## trainX --  They are the samples of my training set. The input shape should be (N_samples, 128,9)        ##
    ## testX -- They are the samples of my test set. The input shpe should be  (N_samples, 128, 9)             ##
    ## 128 is the windows size and 9 is the number of sensors that we have                                     ##
    ##**********************************************Outputs****************************************************##
    ## trainX_norm -- Train set normalized with the min-max                                                    ##
    ## testX_norm -- Test set normalized with the min-max (this min-max must be the one from the training set) ##
    ##*********************************************************************************************************##
    trainX_reshape = np.transpose(trainX,(2,0,1))
    testX_reshape = np.transpose(testX,(2,0,1))
    trainX_norm = np.zeros(trainX.shape)
    testX_norm = np.zeros(testX.shape)
    max_channel = []
    min_channel = []
    for a in range(trainX_reshape.shape[0]):
        max_channel.append(np.amax(np.amax(trainX_reshape[a,:,:], axis= 1)))
        min_channel.append(np.amin(np.amin(trainX_reshape[a,:,:], axis= 1)))
    for a in range(trainX_reshape.shape[0]):
        trainX_norm[:,:,a] = (trainX[:,:,a]-min_channel[a])/(max_channel[a]-min_channel[a])
        testX_norm[:,:,a] = (testX[:,:,a]-min_channel[a])/(max_channel[a]-min_channel[a])
    return trainX_norm, testX_norm


    
    