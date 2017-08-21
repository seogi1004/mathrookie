import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import math

nb_classes = 20
x_classes = 11

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]

def createMatrixData(fileName):
    data = pd.read_csv(fileName)
    count = len(data["시간"])

    # fd = datetime.datetime.strptime(trainData["시간"][0], "%Y/%m/%d %H:%M")
    # print(len(trainData["시간"]))
    # print(fd, fd.isoweekday(), fd.day, fd.hour, fd.minute)

    weekdays = []
    days = []
    hours = []
    minutes = []
    outputs = []

    for i in range(count):
        dateObj = datetime.datetime.strptime(data["시간"][i], "%m/%d/%Y %H:%M")
        weekdays.append(dateObj.isoweekday())
        days.append(dateObj.day)
        hours.append(dateObj.hour)
        minutes.append(dateObj.minute)

        level = math.floor(data["동시 사용자"][i] / 20)
        if(level >= nb_classes):
            level = nb_classes - 1

        outputs.append(level)

    data["구간"] = pd.Series(outputs, index=data.index)
    data["요일"] = pd.Series(weekdays, index=data.index)
    data["일"] = pd.Series(days, index=data.index)
    data["시"] = pd.Series(hours, index=data.index)
    data["분"] = pd.Series(minutes, index=data.index)

    cols = data.columns.tolist()
    cols = cols[-4:] + cols[1:-4]

    data = data[cols]
    print(data.head(1))

    result = data.as_matrix()
    x_data = result[:, 0:-2]

    # 동시 사용자는 제외, 구간만 결과값
    y_data = result[:, [-1]]
    one_hot_data = []

    for value in y_data:
        one_hot_data.append(
            one_hot_encode([ int(value[0]) ], nb_classes)[0]
        )

    return x_data, np.array(one_hot_data)

trainData = createMatrixData("data/aries_train2.csv")
testData = createMatrixData("data/aries_test2.csv")

x_data = trainData[0]
y_data = trainData[1]

# input place holders
X = tf.placeholder(tf.float32, [None, x_classes])
Y = tf.placeholder(tf.int32, [None, nb_classes])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([x_classes, 60]))
b1 = tf.Variable(tf.random_normal([60]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([60, 20]))
b2 = tf.Variable(tf.random_normal([20]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([20, nb_classes]))
b3 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    # Test model and check accuracy
    print('Accuracy:', sess.run(accuracy, feed_dict={X: testData[0], Y: testData[1]}))
