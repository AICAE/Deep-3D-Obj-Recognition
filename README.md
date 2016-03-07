# 3D Object Recognition with Deep Networks
This is the 3D Object Recognition with Deep Networks Project for the 3D Vision course at ETHZ
  
## What is needed:  
* 3D CAD data (Object File Format) to Voxel Data
  * [ModelNet10 - Zip Datei](http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip)
  * [ModelNet40 - Zip Datei](http://modelnet.cs.princeton.edu/ModelNet40.zip)
  * Files can be view in MeshLab
  * Matlab Function to read .OFF files ([Matlab File](http://vision.princeton.edu/pvt/RenderMe/RenderDepth/offLoader.m)) 
* *optional* 2.5D Reconstruction (combine multiple 2.5D Representation into new 3D Representation)
* *optional* 2.5D & 3D Point Cloud Data to Voxel Data (Project Tango & Extra Training Data Sets)
* VoxNet  
  * Convolutional Neural Network
* 3D Shape Net ([3D ShapeNet - Source Code Matlab - Zip](http://vision.princeton.edu/projects/2014/3DShapeNets/3DShapeNetsCode.zip))
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
