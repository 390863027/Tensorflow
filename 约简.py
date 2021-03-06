import tensorflow as tf
sess=tf.InteractiveSession()
#使用tf.InteractiveSession()来构建会话的时候，
#我们可以先构建一个session然后再定义操作（operation），
#如果我们使用tf.Session()来构建会话我们需要在会话构建之前定义好全部的操作（operation）
#然后再构建会话。

#约减---降维

x=tf.constant([[1,2,3],
               [3,2,1],
               [-1,-2,-3]])
boolean_tensor=tf.constant([[True,False,True],
                            [False,False,True],
                            [True,False,False]])
# 乘积方式降维，reduction_indices=1在行内计算，reduction_indices=0在列内计算
print(tf.reduce_prod(x,reduction_indices=1).eval())#行内相乘

print(tf.reduce_min(x,reduction_indices=1).eval())#行内最小值

print(tf.reduce_max(x,reduction_indices=1).eval())#行内最大值

print(tf.reduce_mean(x,reduction_indices=1).eval())#行内平均值

print(tf.reduce_all(boolean_tensor,reduction_indices=1).eval())#行内全部为真，则为真

print(tf.reduce_any(boolean_tensor,reduction_indices=1).eval())#行内存在真，则为真