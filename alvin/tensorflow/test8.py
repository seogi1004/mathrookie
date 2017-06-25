import tensorflow as tf

x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1', dtype=tf.float32)
b1 = tf.Variable(tf.random_normal([2]), name='bias1', dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(W1))
    print(sess.run(b1))