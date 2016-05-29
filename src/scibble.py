from recognizer_voxnet import load_pc, detector_voxnet
import time
#from mayavi import mlab

tic = time.time()
obj_det = detector_voxnet("data/weights_modelnet40_acc0-8234_2016-5-20-3-47.h5")
tictoc = time.time() - tic
print("loading the Detector took {0}".format(tictoc))

np_pc = load_pc("data/deskXYZ.mat")
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