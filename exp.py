import tensorflow as tf

a = tf.reshape(tf.range(27),[3,3,3])
b = tf.reshape(tf.range(27),[3,3,3])
print(a)
print(b)
c = tf.einsum('hwk,hwk->hw',a,b)
print(c)