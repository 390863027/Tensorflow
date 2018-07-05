import tensorflow as tf
sess=tf.Session()
filename_queue=tf.train.string_input_producer(tf.train.match_filenames_once("./blue_jay.jpg"))
reader=tf.WholeFileReader()
key,value=reader.read(filename_queue)
image=tf.image.decode_jpeg(value)
flipImageUpDown=tf.image.encode_jpeg(tf.image.flip_up_down(image))
flipImageLeftRight=tf.image.encode_jpeg(tf.image.flip_left_right(image))
tf.initialize_all_variables().run(session=sess)
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(coord=coord,sess=sess)
example=sess.run(flipImageLeftRight)
print(example)
file=open("flippedUpDown.jpg","wb+")
file.write(flipImageUpDown.eval(session=sess))
file.close()
file=open("flippedLeftRight.jpg","wb+")
file.write(flipImageLeftRight.eval(session=sess))
file.close()