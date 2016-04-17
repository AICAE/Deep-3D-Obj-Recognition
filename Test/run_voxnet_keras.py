from Source.voxnet_keras import lib_IO_hdf5
from Source.voxnet_keras.config import model_cfg
from Source.voxnet_keras import model_keras
import logging

logging.basicConfig(filename='debug.log', level=logging.DEBUG)

voxnet = model_keras.model_vt()
loader = lib_IO_hdf5.Loader_hdf5("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/testing.hdf5",
                                 set_type= "train",
                                 batch_size= 12,
                                 num_batches= 10,
                                 shuffle=True,
                                 valid_split=0.15,
                                 mode="train")
voxnet.fit(generator=loader.generator(),
           samples_per_epoch=loader.return_samples_per_epoche(),
           nb_epoch=1)
# v.load_weights("weightsm")
feature_valid, labels_valid = loader.return_valid_set()
#voxnet.evaluate(feature_valid, labels_valid)
