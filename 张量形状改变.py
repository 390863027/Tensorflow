import tensorflow as tf
sess=tf.InteractiveSession()
x=tf.constant([[2,5,3,-5],
               [0,3,-2,5],
               [4,3,5,3],
               [6,1,4,0]])

print(tf.shape(x).eval())

print(tf.size(x).eval())

print(tf.rank(x).eval())

print(tf.reshape(x,[8,2]).eval())

print(tf.squeeze(x).eval())

print(tf.expand_dims(x,1).eval())