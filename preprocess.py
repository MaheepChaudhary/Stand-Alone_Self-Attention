#preprocess
from config import *

class_names =  ['Jackets','Sweaters', 'Caps','Dresses','Trousers','Shorts', 
                'Sunglasses', 'Sports Shoes', 'Shirts','Tshirts'
               ]
#preprocess

class preprocessing():
    
    def __init__(self,file_name):
        self.file_name = file_name
    
    def pickle_load(self):
        f = pickle.load(open(self.file_name,'rb'))
        return f

    def process(self): 
        f = self.pickle_load()
        ignored_objects = [0,4,5,9,10,11,14]
        train_x,train_y = [],[]
        for i in range(len(f)):
          k = f[i][1]  
          if int(k) in ignored_objects:
            pass
          else:
            if k < 4:
              k-=1
            elif k>=4 and k < 5:
              k-=2
            elif  k>=5 and k< 9:
              k-=3
            elif  k>=9 and k< 10:
              k-=4
            elif  k>=10 and k< 11:
              k-=5 
            elif  k>=11 and k< 14:
              k-=6
            elif k >= 14:
              k-=7     
            train_x.append((np.resize(np.array(tf.keras.preprocessing.image.img_to_array(f[i][0])),[60,60,3])-127.5)/127.5)
            #print("The type of the f[i][1] is {} and the value is {} ".format(type(f[i][1]),f[i][1]))
            train_y.append(tf.one_hot(k,10))
        return np.array(train_x),(np.array(train_y)) 

x = preprocessing('/content/drive/My Drive/Reminder2/Fashion_augmented_dataset/fashion_data_class.pickle')
#x = preprocessing('./data/Object_Detection/fashion_dataset/fashion_data_class.pickle')
train_x,train_y = x.process()
#print("The shape of the train_x is ",train_x.shape)

y = [np.argmax(x) for x in train_y]
