from config import *
from model import *
from preprocess import *

#kf = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 1)
sss = StratifiedKFold(n_splits=5, shuffle = True, random_state=0)
#train_x = np.array([[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]],[[8]],[[9]],[[10]]])
'''
train_x = np.ones([10,96,96,1])
train_y = np.array([0,1,0,0,0,1,1,1,1,0])
train_y = tf.one_hot(train_y,2)
classes = 2
'''

#image_shape = train_x[1:]
classes = 10
epochs = 2
batch_size = 16

image_shape = train_x[0].shape
print("The shape of the image is ",image_shape)

img = Input(shape = image_shape)
resnet = ResNet(classes ,image_shape)(img)
resnet = tk.models.Model(inputs = img,outputs = resnet)
print(resnet.summary())
tk.utils.plot_model(resnet,to_file='model.png')
opt = tk.optimizers.Adam(learning_rate=0.0001)
resnet.compile(optimizer = opt,loss = 'categorical_crossentropy',metrics=['acc'])
val_accuracy,accuracy,loss,validation_loss = [],[],[],[]
#for i in range(epochs):
#|print("The splitting shape is the ",kf.split(train_x))
#for _ in range(epochs):
for train, test in sss.split(train_x,y):
  history = resnet.fit(train_x[train],train_y[train],batch_size = batch_size,validation_data = (train_x[test],train_y[test]),epochs = epochs)
  val_accuracy.extend(history.history['val_acc'])
  accuracy.extend(history.history['acc'])
  loss.extend(history.history['loss'])
  validation_loss.extend(history.history['val_loss'])
