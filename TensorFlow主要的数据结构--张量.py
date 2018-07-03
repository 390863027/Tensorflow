#-*-coding:utf-8-*-

#1，引入Tensorflow模块
import tensorflow as tf
sess=tf.Session()
a=tf.constant(2)
b=tf.constant(5)
print(sess.run(a+b))

#2.创建张量，并获取其元素
import tensorflow as tf
sess=tf.Session()
tens1=tf.constant([[[1,2],[2,3]],[[3,4],[5,6]]])
print(sess.run(tens1)[1,1,0])

#3.创建新的张量
import tensorflow as tf
import numpy as np
x=tf.constant(np.random.rand(32).astype(np.float32))
y=tf.constant([1,2,3])

#4.numpy数组转化成张量
import tensorflow as tf
import numpy as np
sess=tf.Session()
x_data=np.array([[1.,2.,3.],[3.,2.,6.]]) # 2*3matrix
x=tf.convert_to_tensor(x_data,dtype=tf.float32)
print(x)