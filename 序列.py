import tensorflow as tf
sess=tf.InteractiveSession()
x=tf.constant([[2,5,3,-5],
               [0,3,-2,5],
               [4,3,5,3],
               [6,1,4,0]])
listx=tf.constant([1,2,3,4,5,6,7,8])
listy=tf.constant([4,5,8,9])

boolx=tf.constant([[True,False],[False,True]])

print(tf.argmin(x,1).eval()) #列的最大值的位置
print(tf.argmax(x,1).eval()) #行的最小值的位置

#print(tf.listdiff(listx,listy)[0].eval())

print(tf.where(boolx).eval()) #显示正确值

print(tf.unique(listx)[0].eval()) #列表中的唯一值