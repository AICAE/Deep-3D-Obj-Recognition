from keras.models import Sequential
from keras.layers import Convolutional3D, MaxPooling3D, Dense, Dropout, LeakyReLU


#TODO add for all parameters
#Find proper Optimizer
class model_vt
    #initiate Model according to voxnet paper
    def __init__(self):
        self.mdl = Sequential()
        #use shape
        #Convolution1
        self.mdl.add(Convolutional3D())
        self.mdl.add(LeakyReLU())
        #Dropout1
        self.mdl.add()
        #Convolution2
        self.mdl.add(Convolutional3D())
        self.mdl.add(LeakyReLU())
        #MaxPool1
        self.mdl.add(MaxPooling3D())
        #Dropout1
        self.mdl.add(Dropout())
        #Dense1
        self.mdl.add(Dense())
        #Dropout2
        self.mdl.add(Dropout())
        #Dense2
        self.mdl.add(Dense())
        #Compile Model
        self.mdl.compile()

    def fit(self, X_train, y_train, cv=10):
        #TODO add cross-validation
        self.mdl.fit()

    def add_data(self):
        #TODO load old weigths
        #TODO improve fit with old weigths
        self.mdl.fit()

    def evaluate(self, X_test, y_test):
        #TODO make sure to use score from modelnet40/10 paper
        self.mdl.evaluate(X_test, y_test)

    def predict(self,X_predict):
        #TODO add prediction from PointCloud
        self.mdl.predict(X_predict)
