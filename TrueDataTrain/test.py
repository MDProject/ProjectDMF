import pickle

savePath = './TrueDataTrain/TrueParam_mnist'
f = open(savePath,'rb')
param = pickle.load(f)
p = param[1].reshape(param[1].shape[0],1)
print(p.shape)
print('sss')