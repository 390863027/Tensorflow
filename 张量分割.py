import tensorflow as tf
sess=tf.InteractiveSession()
seg_ids=tf.constant([0,1,1,2,2])#下标
tens1=tf.constant([[2,5,3,-5],
                   [0,3,-2,5],
                   [6,1,4,0],
                   [6,1,4,0]])

print(tf.segment_sum(tens1,seg_ids).eval())
print(tf.segment_prod(tens1,seg_ids).eval())
print(tf.segment_min(tens1,seg_ids).eval())
print(tf.segment_max(tens1,seg_ids).eval())
print(tf.segment_mean(tens1,seg_ids).eval())