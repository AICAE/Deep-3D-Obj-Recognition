import numpy as np
from matplotlib import pyplot as plt

from Source.voxnet_keras import lib_IO
from Source.voxnet_keras.config import model_cfg

fname_testset = '/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10_test.tar'
fname_trainset = '/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10_train.tar'

cfg = model_cfg.voxnet_cfg

dataset, labels = lib_IO.load_and_format(fname_testset, cfg)

print('dataset shape is {1}   -- should be .. x 1 x 32 x 32 x 32'.format(dataset.shape))
print('dataset shape is {1}   -- should be .. x 2'.format(labels.shape))

firstimg = dataset[np.random.randint(10,100),1,:,:,:]
print(firstimg.shape)

fig = plt.figure()

