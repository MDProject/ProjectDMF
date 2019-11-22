from FullConnectedNet import *
from LinkFCCNN import *

net_struct = [2*13*13,240,128,64,10]
net = FCNet(net_struct,detailed_info = True)
lengthOfLayers = len(net_struct)

# feed forward [lengthOfLayers-1] steps
for l in range(lengthOfLayers-1):
    net.UpdateDimensionality()
    net.UpdateDeltaTensor()
    net.IterateOneLayer()
net.UpdateDimensionality()

# print out the dimensionality info
net.PrintDimInfo()
