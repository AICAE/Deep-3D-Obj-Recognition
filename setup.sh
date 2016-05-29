#update dist
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y update
sudo apt-get -y upgrade
#requirements
sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic

#libs
sudo apt-get install -y libopenblas-dev liblapack-dev libblas-dev libhdf5-dev libprotobuf-dev libatlas-dev libatlas3gf-base libncurses-dev 

#python tools
sudo apt-get install -y python-dev python-setuptools python-pip python-nose python-numpy python-scipy python-pandas python-h5py python-yaml python-six python-protobuf python-sklearn libzmq mayavi

sudo pip install --upgrade pip jupyter ipython notebook

sudo pip install --upgrade git+https://github.com/scikit-learn/scikit-learn.git

sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

sudo pip install --upgrade --no-deps git+git://github.com/fchollet/keras.git

sudo apt-get -y update
sudo apt-get -y upgrade


