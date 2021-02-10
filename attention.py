from config import *


class Attention(tk.layers.Layer):
    
    def __init__(self,input_channels,output_channel,kernel_size,groups):
        super().__init__()
        self.input_channels = input_channels
        self.output_channel = output_channel    
        self.kernel_size = kernel_size
        self.stride = 1
        self.groups = groups

        assert output_channel % groups == 0
        
        self.rel_h = tk.backend.variable(lambda : tk.backend.truncated_normal((1,1,kernel_size,1,output_channel//2),stddev = 0.1)) 
        #output_channels//2 is the number of channels on which the relative position will be considered,1 denotes the number of those filters and the one after that too and (kernel_size,1) denotes the size of that filter
        self.rel_w = tk.backend.variable(lambda : tk.backend.truncated_normal((1,1,1,kernel_size,output_channel//2),stddev = 0.1)) 

        self.key_weights = Conv2D(self.output_channel,kernel_size = 1,strides = self.stride)
        self.query_weights = Conv2D(self.output_channel,kernel_size = 1,strides = self.stride)
        self.value_weights = Conv2D(self.output_channel,kernel_size = 1,strides = self.stride)

    def call(self,x):
        
        batch,height,width,channels = x.shape
        x_padded = ZeroPadding2D(padding=(self.kernel_size//2,self.kernel_size//2))(x)
        query = self.query_weights(x)
        value = self.value_weights(x_padded)
        key = self.key_weights(x_padded)
        #key,query and value will have the shape of (batch,height,width,depth)
        keys = tf.image.extract_patches(images = key,sizes = [1,self.kernel_size,self.kernel_size,1],strides = [1,self.stride,self.stride,1],rates = [1,1,1,1], padding = "VALID")
        value = tf.image.extract_patches(images = value,sizes = [1,self.kernel_size,self.kernel_size,1],strides = [1,self.stride,self.stride,1],rates = [1,1,1,1], padding = "VALID")
        no_of_kernels = key.shape[-2] - self.kernel_size + 1
        keys = tf.reshape(keys,shape = (-1,no_of_kernels, no_of_kernels,self.kernel_size,self.kernel_size,self.output_channel))
        key_split_h,key_split_w = tf.split(keys,num_or_size_splits = 2,axis = -1)
        key_with_rel = tk.layers.concatenate([key_split_h + self.rel_h,key_split_w + self.rel_w],axis = -1) 
        
        #reshaping the query and key
        key_with_rel = tf.reshape(key_with_rel,(-1,self.groups,no_of_kernels,no_of_kernels,self.kernel_size*self.kernel_size,self.output_channel//self.groups))
        query  = tf.reshape(query,(-1,self.groups,no_of_kernels,no_of_kernels,1,self.output_channel//self.groups))        
        value = tf.reshape(value,(-1,self.groups,no_of_kernels,no_of_kernels,self.kernel_size*self.kernel_size,self.output_channel//self.groups))
        
        #multiplication  of key and query
        #assert key_with_rel.shape == query.shape        
        key_prod_query = query*key_with_rel
        
        # Now the function is passed through the softmax and is multiplied with the values
        s = Softmax(axis = -2)(key_prod_query)
        y = tf.einsum('bnchwk,bnchwk->bnchk',s,value)
        y = tf.reshape(y,(-1,height,width,self.output_channel))
        return y

class AttentionStem(tk.layers.Layer):

    def __init__(self,input_channels,output_channel,kernel_size,groups):
        super().__init__()
        self.input_channels = input_channels
        self.output_channel = output_channel    
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = 1
        self.m = 4
        self.value_weights = self.create_layers() #helps in execution of the funciton create_layers
               
        self.emb_a = tk.backend.variable(lambda : tk.backend.truncated_normal((output_channel//groups,kernel_size),stddev = 0.1)) 
        #output_channels//2 is the number of channels on which the relative position will be considered,1 denotes the number of those filters and the one after that too and (kernel_size,1) denotes the size of that filter
        self.emb_b = tk.backend.variable(lambda : tk.backend.truncated_normal((output_channel//groups,kernel_size),stddev = 0.1)) 
        self.emb_mix = tk.backend.variable(lambda : tk.backend.truncated_normal((self.m,output_channel//groups),stddev = 0.1))

        self.key_weights = Conv2D(self.output_channel,kernel_size = 1,strides = self.stride)
        self.query_weights = Conv2D(self.output_channel,kernel_size = 1,strides = self.stride)
        self.s = Softmax()

    def create_layers(self):
        layers = [Conv2D(self.output_channel,kernel_size = 1,strides = self.stride) for _ in range(self.m)]
        return layers 

    def call(self,x):

        batch,height,width,channels = x.shape
        x_padded = ZeroPadding2D(padding=(self.kernel_size//2,self.kernel_size//2))(x)
        query = self.query_weights(x)
        value = tf.stack([self.value_weights[i](x_padded) for i in range(self.m)])
        key = self.key_weights(x_padded)
        #all the weights are initialized
        no_of_kernels = key.shape[-2] - self.kernel_size + 1 #it is equivalent to the height or width of the original image  i.e. x
        #dividing the image into different patches
        #value_out = []
        a = tf.image.extract_patches(images = value[0] , sizes = [1,self.kernel_size,\
            self.kernel_size,1], strides = [1,1,1,1],rates = [1,1,1,1],padding = "VALID")
        b = tf.image.extract_patches(images = value[1] , sizes = [1,self.kernel_size,\
            self.kernel_size,1], strides = [1,1,1,1],rates = [1,1,1,1],padding = "VALID")
        c = tf.image.extract_patches(images = value[2] , sizes = [1,self.kernel_size,\
            self.kernel_size,1], strides = [1,1,1,1],rates = [1,1,1,1],padding = "VALID")
        d = tf.image.extract_patches(images = value[3] , sizes = [1,self.kernel_size,\
            self.kernel_size,1], strides = [1,1,1,1],rates = [1,1,1,1],padding = "VALID")    
        
        value_out = tf.stack([a,b,c,d]) 
        value_out = tf.reshape(value_out,shape = (self.m,-1,no_of_kernels, no_of_kernels,self.kernel_size,self.kernel_size,\
                                                  self.output_channel))
        key_out = tf.image.extract_patches(images = key, sizes = [1,self.kernel_size,self.kernel_size,1], strides = [1,1,1,1],\
                                           rates = [1,1,1,1],padding = "VALID")
        key_out = tf.reshape(key_out, shape = (-1,self.groups,no_of_kernels,no_of_kernels,self.output_channel//self.groups,\
                                               self.kernel_size*self.kernel_size)) 
        query = tf.reshape(query, shape = (-1,self.groups,no_of_kernels,no_of_kernels,self.output_channel//self.groups,1))

        #calculating Softmax(p(a,b,m))
        emb_logit_a = tk.backend.dot(self.emb_mix,self.emb_a)   #[4(self.m),7(kernel size)] [4,1,7]
        emb_logit_b = tk.backend.dot(self.emb_mix,self.emb_b)    #[4,7,1]
        emb = tf.expand_dims(emb_logit_a,axis = 2) + tf.expand_dims(emb_logit_b,axis = 1) #[4,7,7]
        emb = self.s(emb)
        emb = tf.reshape(emb,shape = (self.m,1,1,1,self.kernel_size,self.kernel_size,1)) 

        V = emb * value_out #the final_value has been calculated
        V = tk.backend.sum(V,axis = 0)
        V = tf.reshape(V, shape = (-1,self.groups,no_of_kernels,no_of_kernels,self.output_channel//self.groups,self.kernel_size*self.kernel_size)) 
        
        out = Softmax(axis = -1)(key_out*query)
        out = tf.reshape(tf.einsum('bnchwk,bnchwk->bnchw',out,V), shape = (-1,height,width,self.output_channel))
        return out


