from config import *
from attention import *

#Identity block of the resnet architecture in which spatial convolution is replaced by the Attetnion layer 

#Identity block of the resnet architecture in which spatial convolution is replaced by the Attetnion layer 
class resblock(tk.layers.Layer):

    def __init__(self,filters,stride,stem):
        super().__init__()
        self.f1,self.f2 = filters
        self.stride = stride
        self.stem = stem
        self.conv1 = tk.Sequential([
                                    tk.layers.Conv2D(self.f1, kernel_size = (1,1)),
                                    tk.layers.BatchNormalization(),
                                    tk.layers.Activation('tanh'),
                                   ])
        if stem:
            self.conv2 = tk.Sequential([
                                        AttentionStem(input_channels = self.f1,output_channel = self.f1,kernel_size = 7,groups =8),
                                        #Conv2D(self.f1,kernel_size = 7,strides = 1,padding = 'same'),
                                        BatchNormalization(),
                                        tk.layers.Activation('tanh'),
                                      ])
        else:
            self.conv2 = tk.Sequential([
                                        Attention(input_channels = self.f1,output_channel = self.f1,kernel_size = 7,groups =8),
                                        #Conv2D(self.f1,kernel_size = 7,strides = 1,padding = 'same'),
                                        BatchNormalization(),
                                        tk.layers.Activation('tanh'),
                                      ])

        self.conv3 = tk.Sequential([
                                   Conv2D(self.f2, kernel_size = (1,1)),
                                   BatchNormalization(),
                                   tk.layers.Activation('tanh'),
                                  ])
        
        self.shortcut = tk.Sequential([
                                      Conv2D(self.f2, strides = self.stride, kernel_size = 1,use_bias = False),
                                      BatchNormalization(),
                                     ])
        self.ap = AveragePooling2D(pool_size=(self.stride,self.stride),padding = "same")
        self.tanh = tk.layers.Activation('tanh')           

    def call(self,x):
        x_short = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.stride >= 2:
            x = self.ap(x)
        x_short = self.shortcut(x_short)
        x = tf.math.add(x,x_short)
        x = self.tanh(x)
        return x

#Renet model contains the full model which in turns calls *resnet_block* and the *attention layer depending on the execution of the layer, whether it is stem or core layer
class ResNet(tk.layers.Layer):
    
    def __init__(self,classes,shape_of_image):
        super().__init__()
        self.classes = classes
        self.shape_of_image = shape_of_image
        self.zero_pad = ZeroPadding2D(padding=(3,3))
        self.conv = Conv2D(64,kernel_size= (7,7),strides = (2,2))   
        self.bn = BatchNormalization() 
        self.mp = MaxPooling2D(pool_size=(3,3),strides = (2,2))
        self.ap = AveragePooling2D(pool_size=(2,2))
        self.block1 = resblock([64,128],1,True)
        self.block2 = resblock([128,128],1,False)
        self.block3 = resblock([128,256],2,False)
        '''
        self.block4 = resblock([256,256],1,False)
        self.block5 = resblock([256,256],2,False)
        self.block6 = resblock([256,256],1,False)
        self.block7 = resblock([256,512],1,False)
        self.block8 = resblock([512,512],1,False)
        '''
        self.f = Flatten()
        self.d = Dense(self.classes,activation = 'softmax')
        

    def call(self,img):
        x = self.zero_pad(img)
        x = self.conv(x)
        x = self.bn(x)
        x = self.mp(x)
        x = self.block1(x)
        #x = Dropout(0.5)(x)
        x = self.block2(x)
        #x = Dropout(0.3)(x)
        x = self.block3(x)
        #x = Dropout(0.2)(x)
        #x = self.block4(x)
        #x = Dropout(0.3)(x)
        '''
        x = self.block5(x)
        x = self.block6(x)
        x = Dropout(0.3)(x)
        x = self.block7(x)
        x = self.block8(x)
        '''
        x = self.ap(x)
        dense = self.f(x)
        result = self.d(dense)
        return result
                                
#------------------------------------------------------------------------END--------------------------------------------------------------------#
#------------------------------------------------------------------------END--------------------------------------------------------------------#--------------------END--------------------------------------------------------------------#