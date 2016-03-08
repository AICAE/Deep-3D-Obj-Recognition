# 3D Object Recognition with Deep Networks
This is the 3D Object Recognition with Deep Networks Project for the 3D Vision course at ETHZ
  
## What is needed:  
#####Input Data/Voxel/Occupancy Grid
* 3D CAD data (Object File Format) to Voxel Data / Occupancy Grid
  * [ModelNet10 - Zip Datei](http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip)
  * [ModelNet40 - Zip Datei](http://modelnet.cs.princeton.edu/ModelNet40.zip)
  * .OFF Files can be view in MeshLab
  * Matlab Function to read .OFF files ([Matlab File](http://vision.princeton.edu/pvt/RenderMe/RenderDepth/offLoader.m)) 
  * Function in the 3D ShapeNet Source code to transform to occupancy grid is called ...
* *optional* 2.5D Reconstruction (combine multiple 2.5D Representation into new 3D Representation)
* *optional* 2.5D & 3D Point Cloud Data to Voxel Data (Project Tango & Extra Training Data Sets)  

##### VoxNet  
* Convolutional Neural Network
* Input Data
  * Rotation Augementation
  * Multiresolution Input
* Training: Stoastic Gradient Decent with Momentum
  * learning rate = 0.0001
  * momentum parameter = 0.9
  * batch size = 32
  * learning rate decrease: 10 per 40000 batches.
* Dropout Regularization after output of each layer
* Initialization:
  * Convolutional Layers: 
    * forward Propagation: zero mean Gaussian with std.dev = sqrt(2/n_l), n_l = dimension(input array(30x30x30) layer l) * input channels layer l
    * backward Propagation: zero mean Gaussion with std.dev = sqrt(2/n*_l), n*_l = dimension(input array(30x30x30)layer l) * input channels layer l-1 
  * Dense Layers: zero-mean Gaussion with std.dev=0.01  

#####3D Shape Net
* [3D ShapeNet - Source Code Matlab - Zip](http://vision.princeton.edu/projects/2014/3DShapeNets/3DShapeNetsCode.zip)
  * Special Learning Algorithm
  * Convolutional Deep Beliefe Network

## Steps:
1. Get ModelNet 3D CAD Data & Read & Transform *if necessary*
2. Build VoxNet & 3D ShapeNet
3. Train, Validate & Tune
  * Not Sure if *Cross Validation* works with the mix of 3D Training and 2.5D Validation Data, but i guess there is someway
  * Depending on how much parameters are left to tune and how much GPU power we have this might take a while (it took them 2 days for 40 objects on a cluster)
4. Test on Spare Data and try to recreate Experiment from *Wu et al.* & *Maturana et al.*
5. *optional* Maybe try to get some Point Cloud Data from Tango Tablet and test Models against that


## useful:
#### PCL ([www.pointclouds.org](http://pointclouds.org))
* [Documentation](http://pointclouds.org/documentation/)
* [Point Cloud to SuperVoxel](http://pointclouds.org/documentation/tutorials/supervoxel_clustering.php)  
* [PointCloud Python Support(bad)](http://pointclouds.org/news/2013/02/07/python-bindings-for-the-point-cloud-library/)
  
#### TensorFlow ([www.tensorflow.org](https://www.tensorflow.org))
* [API Documentation](https://www.tensorflow.org/versions/r0.7/api_docs/index.html)  
* [Tutorials](https://www.tensorflow.org/versions/r0.7/tutorials/index.html)  
* [TensorFlow GitHub](https://github.com/tensorflow/tensorflow)  

#### MeshLab ([http://meshlab.sourceforge.net/](http://meshlab.sourceforge.net/))
* 
