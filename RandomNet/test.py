from DynamicMeanField import *
from LinkFCCNN import *
from FullConnectedNet import *
import numpy as np

# Full Connect test
struct_fc = [1280,64,10]
weight_fc = None
bias_fc = None
Net_FC = FCNet(struct_fc, weight = weight_fc, bias = bias_fc, input_cor = None, input_m = None, detailed_info = True)
length_FC_Layers = len(struct_fc)

# feed forward [lengthOfLayers-1] steps
for l in range(length_FC_Layers-1):
    Net_FC.UpdateDimensionality()
    Net_FC.UpdateDeltaTensor()
    Net_FC.IterateOneLayer()
Net_FC.UpdateDimensionality()
Net_FC.PrintDimInfo()