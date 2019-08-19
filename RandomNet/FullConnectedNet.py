import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

# this Moledule could be either used independently for Full Connected network
class FCNet:

    ############# PARAM #############
    g_w = 0.8   # weight
    rho = 0.1
    sigma_b = 0.1   # bias
    ############# PARAM #############
    
    # struct is a list which consists of each layer's neurons number 
    # [input_cor] is None/N*N size matrix   [input_mean] is None/N*1 size column vector 
    def __init__(self, struct, phi = np.tanh, weight = None, bias = None, input_cor = None, input_m = None, detailed_info = False, eigen_check = False):
        self.struct = struct
        self.layer_len = len(struct)
        self.phi = phi
        self.detailed_info = detailed_info
        self.eigen_check = eigen_check
        self.current_layer = 0  # initialized at input layer
        if weight is None:
            self.weight = ['0']
            self.InitialWeight()
        else:
            self.weight = weight
        if bias is None:
            self.bias = ['0']
            self.InitialBias()
        else:
            self.bias = bias
        if input_cor is None:
            self.InitialInputCorvarianceTensor(FCNet.rho)
        else:
            if input_cor.shape[0] == self.struct[0]:
                self.C = input_cor
            else:
                print('Input corvariance matrix does not match the network structure')
                sys.exit()
        if input_m is None:
            self.InitialMeanActivation()
        else:
            self.h_mean = input_m
        self.ratio = []
        self.dimList = []
    
    def InitialWeight(self):
        # self.weight is a list ['0',weight(1),weight(2),...] where weight(l) connects layer l and layer l-1
        for l in range(1,self.layer_len):
            weightPerLayer = np.random.randn(self.struct[l],self.struct[l-1])*np.sqrt(FCNet.g_w/self.struct[l-1])
            self.weight.append(weightPerLayer)

    def InitialBias(self):
        # self.bias is a list ['0',bias(1),bias(2),...] where bias(l) belongs to layer l and it's a column vector
        for l in range(1,self.layer_len):
            biasPerLayer = np.random.randn(self.struct[l],1)*np.sqrt(FCNet.sigma_b)
            self.bias.append(biasPerLayer)

    def InitialInputCorvarianceTensor(self,rho):
        input_size = self.struct[0]
        self.C = np.zeros([input_size,input_size])
        # initialize upper diagonal and diagonal parts
        for i in range(input_size):
            for j in range(i,input_size):
                if i == j:
                    self.C[i][j] = 1
                else:
                    self.C[i][j] = (-rho + 2*rho*np.random.rand())/np.sqrt(input_size)
        # keep it symmetric
        for i in range(input_size):
            for j in range(i):
                self.C[i][j] = self.C[j][i]
    
    def InitialMeanActivation(self):
        input_size = self.struct[0]
        self.h_mean = np.zeros([input_size,1])
    
    def UpdateDeltaTensor(self):
        # based on current corvariance matrix 
        lidx = self.current_layer
        self.Delta = np.dot(np.dot(self.weight[lidx+1],self.C),self.weight[lidx+1].T)
    
    def IterateOneLayer(self):
        if self.detailed_info:
            print('Current layer:   {}'.format(self.current_layer))
        next_layer_size = self.struct[self.current_layer+1]
        h_mean_next = np.zeros([next_layer_size,1])
        C_next = np.zeros([next_layer_size,next_layer_size])
        x = np.random.randn(1,20000)
        y = np.random.randn(1,20000)
        for i in range(next_layer_size):
            h_mean_next_pre = np.sqrt(self.Delta[i][i])*x + np.dot(self.weight[self.current_layer+1],self.h_mean)[i][0] + self.bias[self.current_layer+1][i][0]
            phi_pre = self.phi(h_mean_next_pre)
            h_mean_next[i] = phi_pre.sum()/20000
        # update corvariance matrix
        for i in range(next_layer_size):
            for j in range(next_layer_size):
                Phi1 = self.phi(np.sqrt(self.Delta[i][i])*x + np.dot(self.weight[self.current_layer+1],self.h_mean)[i][0] + self.bias[self.current_layer+1][i][0])
                PHI = self.Delta[i][j]/np.sqrt(self.Delta[i][i]*self.Delta[j][j])
                Phi2 = self.phi(np.sqrt(self.Delta[j][j])*(PHI*x+np.sqrt(1-PHI*PHI)*y) + np.dot(self.weight[self.current_layer+1],self.h_mean)[j][0] + self.bias[self.current_layer+1][j][0])
                if 1-PHI*PHI < 0:
                    print('negtive root error')
                A = (Phi1*Phi2).sum()/20000
                B = self.h_mean[i]*self.h_mean[j]
                C_next[i][j] = A - B
        # update the next layer information to current layer
        self.h_mean = h_mean_next.copy()
        self.C = C_next.copy()
        self.current_layer = self.current_layer + 1

    def CorEigenSignCheck(self):
        # check the ratio of the negtive eigenvalue, called before iterating ahead
        num_current_neuron = self.struct[self.current_layer]
        eigenvalue, eigenvector = np.linalg.eig(self.C)
        minus_num = 0
        for n in range(eigenvalue.shape[0]):
            if eigenvalue[n] < 0:
                minus_num = minus_num + 1
        self.ratio.append(minus_num/eigenvalue.shape[0])
        
    # extract the dimensionality of current layer
    def UpdateDimensionality(self):
        Ctrace = np.trace(self.C)
        N = self.struct[self.current_layer]
        dim = Ctrace*Ctrace/np.trace(np.dot(self.C,self.C))/N
        self.dimList.append(dim)
        
    # print out dimensionality information layer by layer
    def PrintDimInfo(self):
        for l in range(self.layer_len):
            print('FC layer {0} has dimensionality {1}'.format(l,self.dimList[l]))   

# API linked to TrueDataTrain
# Load the trained parameters and transform it to the format required by "FCNet" class
# return [struct,weight,bias] as constructor's parameter
def FCReadParam(fpath):
    f = open(fpath,'rb')
    ParamList = pickle.load(f)
    # get the layers of CNN part and layer's structure
    lenOfCNN = 0
    lenOfFC = 0
    lenOfNet = len(ParamList)//2 + 1    # input layer is included
    struct_cnn = [0]
    struct_fc = [0]
    for i in range(len(ParamList)):
        # i increase by 2 due to interleave structure of ParamList
        if ParamList[i].ndim > 2:
            lenOfCNN += 1
            struct_cnn.append((ParamList[i].shape)[0])
        if ParamList[i].ndim == 2:
            struct_fc.append((ParamList[i].shape)[0])
    # There are total 2 cases: I. has CNN parts II. does not have CNN parts
    # I. at least 1(2 if last CNN layer is counted) FC layer
    # II. FC net 
    fc_startIdx = 0 # first weight index in ParamList
    if lenOfCNN > 0:
        lenOfCNN += 1 
        fc_startIdx = (lenOfCNN-1)*2
        struct_cnn[0] = (ParamList[0].shape)[1]
        struct_fc[0] = (ParamList[fc_startIdx].shape)[1] # last layer of CNN should be counted
        lenOfFC = lenOfNet - lenOfCNN + 1   # last layer of CNN should be counted
    else:   # net does not have CNN structure
        lenOfFC = lenOfNet  # input layer is included
        struct_fc[0] = (ParamList[0].shape)[1]
    # reconstruct weight
    weight = ['0']
    bias = ['0']
    for n in range(lenOfFC-1):
        weight.append(ParamList[fc_startIdx+2*n])
        bias.append(ParamList[fc_startIdx+2*n+1].reshape(ParamList[fc_startIdx+2*n+1].shape[0],1))
    return struct_fc, weight, bias




