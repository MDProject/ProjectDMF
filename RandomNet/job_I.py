from DynamicMeanField import *
from LinkFCCNN import *
from FullConnectedNet import *
import pickle

fpath_mnist = './ProjectDMF/TrueDataTrain/TrueParam_mnist'
fpath_cifar10 = './ProjectDMF/TrueDataTrain/TrueParam_cifar10'
fpath_mnist_param = './ProjectDMF/TrueDataTrain/TrueDataCorvariance'

# true corvariance - trained parameters for MNIST
fp = open(fpath_mnist_param,'rb')
struct_cnn = [1,3,3,2]
weight_cnn = None
bias_cnn = None
h_mean = None
Input_CTensor = None
length_CNN_Layers = len(struct_cnn)

Net_CNN = DMFNet(struct_cnn,6,input_size = 28, weight = weight_cnn, bias = bias_cnn, input_CTensor = Input_CTensor, input_h_mean = h_mean, detailed_info = True, eigen_check = False)
for l in range(length_CNN_Layers-1):
    Net_CNN.UpdateDimensionality()
    #net.PrintCorvarianceHeatMap()
    Net_CNN.UpdateDeltaTensor() 
    Net_CNN.IterateOneLayer()
Net_CNN.UpdateDimensionality()


mean_fc,cor_fc = LinkCNN2FC(Net_CNN)
struct_fc = [mean_fc.shape[0],64,10]
weight_fc = None
bias_fc = None
Net_FC = FCNet(struct_fc, weight = weight_fc, bias = bias_fc, input_cor = cor_fc, input_m = mean_fc, detailed_info = True)
length_FC_Layers = len(struct_fc)

# feed forward [lengthOfLayers-1] steps
for l in range(length_FC_Layers-1):
    Net_FC.UpdateDimensionality()
    Net_FC.UpdateDeltaTensor()
    Net_FC.IterateOneLayer()
Net_FC.UpdateDimensionality()

Net_CNN.PrintDimInfo()
Net_FC.PrintDimInfo()