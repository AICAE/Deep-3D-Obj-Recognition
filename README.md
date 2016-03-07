# 3D Object Recognition with Deep Networks
This is the 3D Object Recognition with Deep Networks Project for the 3D Vision course at ETHZ
  
##What is needed:  
* 3D CAD data to Voxel Data (Traning Data by WU et al.) -*if ModelNet DataSet is not voxel data* 
* 2.5D & 3D Point Cloud Data to Voxel Data (Project Tango & Extra Training Data Sets)
  * 2.5D Reconstruction (maybe need algorithm to figure out what voxels are observed and which unknown from point cloud data) 
* VoxNet  
  * Convolutional Neural Network
* 3D Shape Net  
  * Special Learning Algorithm
  * Convolutional Deep Beliefe Network

##Steps
1. Get ModelNet 3D CAD Data & Transform(if neccessary)
2. Build VoxNet & 3D ShapeNet
3. Train, Validate & Tune
  * Not Sure if Tensorflow has *Cross Validation* but i would guess so
  * Depending on how much parameters are left to tune and how much GPU power we have this might take a while
4. Test on Spare Data and try to recreate Experiment from *Wu et al.* & *Maturana et al.*
5. Maybe try to get some Point Cloud Data from Tango Tablet and test Models against that


## What can be used:
#### PCL ([www.pointclouds.org](http://pointclouds.org))
* [Documentation](http://pointclouds.org/documentation/)
* [Point Cloud to SuperVoxel](http://pointclouds.org/documentation/tutorials/supervoxel_clustering.php)  
* [PointCloud Python Support(bad)](http://pointclouds.org/news/2013/02/07/python-bindings-for-the-point-cloud-library/)
  
#### TensorFlow ([www.tensorflow.org](https://www.tensorflow.org))
* [API Documentation](https://www.tensorflow.org/versions/r0.7/api_docs/index.html)  
* [Tutorials](https://www.tensorflow.org/versions/r0.7/tutorials/index.html)  
* [TensorFlow GitHub](https://github.com/tensorflow/tensorflow)  
  
