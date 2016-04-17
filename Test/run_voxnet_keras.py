from Source.voxnet_keras import lib_IO_hdf5
from Source.voxnet_keras.config import model_cfg
from Source.voxnet_keras import model_keras


voxnet = model_keras.model_vt()
loader = lib_IO_hdf5.Loader_hdf5("/home/tg/Projects/Deep-3D-Obj-Recognition/Source/Data/testing.hdf5",
                                 set_type= "train",
                                 batch_size= 12 * 64,
                                 num_batches=2,
                                 shuffle=True,
                                 valid_split=0.15)
voxnet.fit(generator=model_keras.FitGenerator(loader = loader), samples_per_epoch=16 * 266, nb_epoch=40)
# v.load_weights("weightsm")
feature_valid, labels_valid = loader.return_valid_set()
voxnet.evaluate(feature_valid, labels_valid)
