import tensorflow as tf
from alvin.aries.cpu_utility import get_matrix_data

ln_count = 100
nb_classes = 100
model_path = "cpu_softmax.ckpt"

trainData = get_matrix_data("data/cpu_train.csv", nb_classes)
testData = get_matrix_data("data/cpu_test.csv", nb_classes)

x_data = trainData[0]
y_data = trainData[1]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph("model/" + model_path + ".meta")
saver.restore(sess, tf.train.latest_checkpoint("model/"))

inputs = tf.get_collection("input")
X = inputs[0]
Y = inputs[1]
train_op_final = tf.get_collection('optimizer')[0]
prediction = tf.get_collection('prediction')[0]
loss = tf.get_collection('cost')[0]
is_training = tf.get_collection('is_training')[0]
accuracy = tf.get_collection('accuracy')[0]

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
