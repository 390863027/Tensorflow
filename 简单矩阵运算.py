import tensorflow as tf
sess=tf.InteractiveSession()# tf.InteractiveSession()默认自己就是用户要操作的session，
# 而tf.Session()没有这个默认，因此用eval()启动计算时需要指明session。
x=tf.constant([[2,5,3,-5],
               [0,3,-2,5],
               [4,3,5,3],
               [6,1,4,0]])
y=tf.constant([[4,-7,4,-3,4],
               [6,4,-7,4,7],
               [2,3,2,1,4],
               [1,5,5,5,2]])
floatx=tf.constant([[2.,5.,3.,-5.],
                    [0.,3.,-2.,5.],
                    [4.,3.,5.,3.],
                    [6.,1.,4.,0.]])
print(tf.transpose(x).eval()) # 转置

print(tf.matmul(x,y).eval()) # 矩阵乘法

print(tf.matrix_determinant(floatx).eval()) # 矩阵行列式

print(tf.matrix_inverse(floatx).eval()) # 矩阵逆置

print(tf.matrix_solve(floatx,[[1],[1],[1],[1]]).eval()) # 求解矩阵