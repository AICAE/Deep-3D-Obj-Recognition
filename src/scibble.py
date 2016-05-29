#!/usr/bin/python3

from recognizer_voxnet import load_pc, detector_voxnet
import time
#from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot():
	np_pc = load_pc("data/chairXYZ.mat")
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(np_pc[:,0], np_pc[:,1], np_pc[:,2])
	plt.show()

def main():
	tic = time.time()
	obj_det = detector_voxnet("data/weights_modelnet40_acc0-8234_2016-5-20-3-47.h5")
	tictoc = time.time() - tic
	print("loading the Detector took {0}".format(tictoc))

	np_pc = load_pc("data/chair.mat")
	#mlab.points3d(np_pc[:,0], np_pc[:,1], np_pc[:,2])
	#mlab.show()


	tic = time.time()
	np_vox = obj_det.voxilize(np_pc)
	tictoc = time.time() - tic
	print("Voxelizing took {0} for {1} points".format(tictoc, np_pc.shape[0]))

	#mlab.points3d(np_vox[:,0], np_vox[:,1], np_vox[:,3], mode='cube')
	#mlab.show

	tic = time.time()
	label = obj_det.predict(X_pred=np_vox)
	tictoc = time.time() - tic

	print("Detection took {0}".format(tictoc))
	print("Wuee a " + label + " was detected")



plot()





