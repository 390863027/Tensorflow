import tensorflow as tf
sess=tf.InteractiveSession()
t_matrix=tf.constant([[1,2,3],
                      [4,5,6],
                      [7,8,9]])
t_array=tf.constant([1,2,3,4,9,8,6,5])
t_array2=tf.constant([2,3,4,5,6,7,8,9])

print(tf.slice(t_matrix,[1,1],[2,2]).eval()) # 张量切片

#tf.split(0,2,t_array)

print(tf.tile([1,2],[3]).eval()) # 平铺小张量三次

print(tf.pad(t_matrix,[[0,1],[2,1]]).eval())#填充0 [上，下][左，右]

print(tf.stack([t_array,t_array2]).eval())# pack后改成stack

print(sess.run(tf.unstack(t_matrix)))

print(tf.reverse(t_matrix,[False,True]).eval())#[False,True]行逆转，[True，False]行列逆转