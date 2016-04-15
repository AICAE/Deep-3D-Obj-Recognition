from voxnet_keras import lib_IO_hdf5
from  voxnet_keras import model_keras

voxnet = model_keras.model_vt()

for feat, lab in lib_IO_hdf5.loader_hdf5("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/modelnet10.hdf5",
                                 set_type= "train",
                                 batch_size= 12 * 128):
    voxnet.git