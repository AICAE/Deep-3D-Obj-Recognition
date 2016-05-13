import lib_IO_hdf5
from config import model_cfg
import model_keras
import logging
import numpy as np
import sys
import time

tic = time.time()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

dataset_name = "modelnet10"

loader = lib_IO_hdf5.Loader_hdf5_Convert_Np("data/" + dataset_name + ".hdf5",
                                            batch_size= 32,
                                            shuffle=True,
                                            has_rot= False,
                                            valid_split=0.12,
                                            )

voxnet = model_keras.model_vt(nb_classes=loader.return_nb_classes(), dataset_name=dataset_name)

voxnet.fit(generator=loader.train_generator(),
          samples_per_epoch=loader.return_num_train_samples(),
          nb_epoch=80,
          valid_generator= loader.valid_generator(),
          nb_valid_samples = loader.return_num_valid_samples())


voxnet.evaluate(evaluation_generator = loader.evaluate_generator(),
               num_eval_samples=loader.return_num_evaluation_samples())

tictoc = time.time() - tic
print("the run_keras with Conversion to Numpy took {0} seconds".format(tictoc))

tic = time.time()

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# with lib_IO_hdf5.Loader_hdf5("data/testing.hdf5",
#                                  batch_size= 12,
#                                  shuffle=True,
#                                  has_rot= False,
#                                  valid_split=0.15,
#                             ) as loader:
#     #Initiate Model
#     voxnet = model_keras.model_vt(nb_classes=loader.return_nb_classes())
#     #train model
#     voxnet.fit(generator=loader.train_generator(),
#               samples_per_epoch=loader.return_num_train_samples(),
#               nb_epoch=2,
#               valid_generator= loader.valid_generator(),
#               nb_valid_samples = loader.return_num_valid_samples())
#     #evaluate model
#     voxnet.evaluate(evaluation_generator = loader.evaluate_generator(),
#                    num_eval_samples=loader.return_num_evaluation_samples())
#
# tictoc = time.time() - tic
# print("the run_keras without Conversion to Numpy run took {0} seconds".format(tictoc))


# def gen():
#     while 1:
#         feat = np.random.randint(0,1,[12,1,32,32,32])
#         lab = np.random.randint(1,3,[12,])
#         yield feat, lab
#
# def valid_gen():
#     while 1:
#         feat = np.random.randint(0,1,[12,1,32,32,32])
#         lab = np.random.randint(1,3,[12,])
#         yield feat, lab
#
# def eval_gen():
#     while 1:
#         feat = np.random.randint(0,1,[12,1,32,32,32])
#         lab = np.random.randint(4,5,[12,])
#         yield feat, lab
#
#
# voxnet = model_keras.model_vt()
# voxnet.fit(generator=gen(),
#           samples_per_epoch=12 * 10,
#           nb_epoch=4,
#           valid_generator= valid_gen(),
#           nb_valid_samples = 12 * 3)
#
#
# voxnet.evaluate(evaluation_generator = eval_gen(),
#                num_eval_samples= 12 * 4)
