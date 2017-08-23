import tensorflow as tf
from alvin.aries.cpu_utility import get_matrix_data
from alvin.aries.cpu_utility import batch_norm_layer

ln_count = 1000
ln_rate = 0.01
x_classes = 6
h_layer_1 = 40
h_layer_2 = 40
nb_classes = 100
model_path = "cpu_softmax.ckpt"

trainData = get_matrix_data("data/cpu_train.csv", nb_classes)
testData = get_matrix_data("data/cpu_test.csv", nb_classes)

x_data = trainData[0]
y_data = trainData[1]

# input place holders
X = tf.placeholder(tf.float32, [None, x_classes])
Y = tf.placeholder(tf.int32, [None, nb_classes])
is_training = tf.placeholder(tf.bool)

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([x_classes, h_layer_1]))
b1 = tf.Variable(tf.random_normal([h_layer_1]))
W2 = tf.Variable(tf.random_normal([h_layer_1, h_layer_2]))
b2 = tf.Variable(tf.random_normal([h_layer_2]))
W3 = tf.Variable(tf.random_normal([h_layer_2, nb_classes]))
b3 = tf.Variable(tf.random_normal([nb_classes]))

# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(X, W1), b1)
layer_1 = batch_norm_layer(layer_1, is_training=is_training, scope='layer_1_bn')
layer_1 = tf.nn.relu(layer_1)
# Hidden layer with RELU activation
layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
layer_2 = batch_norm_layer(layer_2, is_training=is_training, scope='layer_2_bn')
layer_2 = tf.nn.relu(layer_2)
# Output layer with linear activation
out_layer = tf.matmul(layer_2, W3) + b3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=Y))
train_step = tf.train.AdamOptimizer(ln_rate).minimize(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if update_ops:
    train_ops = [train_step] + update_ops
    train_op_final = tf.group(*train_ops)

# Get accuracy of model
prediction = tf.argmax(out_layer, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# cache data
tf.add_to_collection("input", X)
tf.add_to_collection("input", Y)
tf.add_to_collection("vars", W1)
tf.add_to_collection("vars", b1)
tf.add_to_collection("vars", W2)
tf.add_to_collection("vars", b2)
tf.add_to_collection("vars", W3)
tf.add_to_collection("vars", b3)
tf.add_to_collection("cost", loss)
tf.add_to_collection("prediction", prediction)
tf.add_to_collection("accuracy", accuracy)
tf.add_to_collection("optimizer", train_op_final)
tf.add_to_collection("is_training", is_training)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for step in range(ln_count):
    _, train_accuracy = sess.run([train_op_final, accuracy], feed_dict={X: x_data, Y: y_data, is_training: True})

    if step % 100 == 0:
        _, train_accuracy, train_loss, train_pred = \
            sess.run([train_op_final, accuracy, loss, prediction], feed_dict={X: x_data, Y: y_data, is_training: False})

        print("Step: {:5}\tLoss: {:.3f}\tAccuracy : {:.2%}".format(step, train_loss, train_accuracy), train_pred)

# Calculate accuracy for all mnist test images
print("Test accuracy for the latest result: %g" % accuracy.eval(
    feed_dict={X: testData[0], Y: testData[1], is_training: False}))

# save data
saver.save(sess, "model/" + model_path)
