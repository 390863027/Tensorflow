import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
# %matplotlib inline # matplotlib inline是jupyter notebook里的命令,
# 意思是将那些用matplotlib绘制的图显示在页面里而不是弹出一个窗口。
import matplotlib.pyplot as plt

trainsamples = 200
testsamples = 60

#这里我们将代表模型，一个简单的输入，一个隐藏的sigmoid激活层
def model(X, hidden_weights1, hidden_bias1, ow):
    hidden_layer =  tf.nn.sigmoid(tf.matmul(X, hidden_weights1)+ b)
    return tf.matmul(hidden_layer, ow)

dsX = np.linspace(-1, 1, trainsamples + testsamples).transpose()
dsY = 0.4* pow(dsX,2) +2 * dsX + np.random.randn(*dsX.shape) * 0.22 + 0.8

plt.figure() # 创建新的figure
plt.title('Original data')
plt.scatter(dsX,dsY) #绘制数据点的散点图
X = tf.placeholder("float")
Y = tf.placeholder("float")

hw1 = tf.Variable(tf.random_normal([1, 10], stddev=0.01)) # 创建第一个隐藏图层
ow = tf.Variable(tf.random_normal([10, 1], stddev=0.01)) # 创建输出连接
b = tf.Variable(tf.random_normal([10], stddev=0.01)) # 创建偏差

model_y = model(X, hw1, b, ow) #

cost = tf.pow(model_y-Y, 2)/(2) # 损失函数

train_op = tf.train.AdamOptimizer(0.0001).minimize(cost) # 构造一个优化器
# 在会话中启动图表
with tf.Session() as sess:
    tf.global_variables_initializer().run()  # 初始化所有变量

    for i in range(1, 10):

        trainX, trainY = dsX[0:trainsamples], dsY[0:trainsamples]
        for x1, y1 in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X: [[x1]], Y: y1})
        testX, testY = dsX[trainsamples:trainsamples + testsamples], dsY[0:trainsamples:trainsamples + testsamples]

        cost1 = 0.
        for x1, y1 in zip(testX, testY):
            cost1 += sess.run(cost, feed_dict={X: [[x1]], Y: y1}) / testsamples
            print("Average cost for epoch " + str(i) + ":" + str(cost1))
        dsX, dsY = shuffle(dsX, dsY)  # 随机化样本以实施更好的训练