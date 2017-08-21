import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import math

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

    for i in range(count):
        dateObj = datetime.datetime.strptime(data["시간"][i], "%Y/%m/%d %H:%M")
        weekdays.append(dateObj.isoweekday())
        days.append(dateObj.day)
        hours.append(dateObj.hour)
        minutes.append(dateObj.minute)

    data["요일"] = pd.Series(weekdays, index=data.index)
    data["일"] = pd.Series(days, index=data.index)
    data["시"] = pd.Series(hours, index=data.index)
    data["분"] = pd.Series(minutes, index=data.index)

    cols = data.columns.tolist()
    cols = cols[-4:] + cols[1:-4]
    data = data[cols]
    print(data.head(1))

    result = data.as_matrix()
    x_data = result[:, 0:-1]
    y_data = result[:, [-1]]

    return x_data, y_data

trainData = createMatrixData("data/aries_train.csv")

x_data = trainData[0]
y_data = trainData[1]

# input place holders
X = tf.placeholder(tf.float32, [None, 13])
Y = tf.placeholder(tf.float32, [None, 1])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([13, 20]))
b1 = tf.Variable(tf.random_normal([20]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([20, 10]))
b2 = tf.Variable(tf.random_normal([10]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([10, 1]))
b3 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

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

    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))