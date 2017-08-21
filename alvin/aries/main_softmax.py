import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import math

nb_classes = 20

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
        dateObj = datetime.datetime.strptime(data["시간"][i], "%Y/%m/%d %H:%M")
        weekdays.append(dateObj.isoweekday())
        days.append(dateObj.day)
        hours.append(dateObj.hour)
        minutes.append(dateObj.minute)

        level = math.floor(data["동시 사용자"][i] / 20)
        if(level > nb_classes):
            level = nb_classes

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

    return x_data, y_data

trainData = createMatrixData("data/aries_train.csv")
#
x_data = trainData[0]
y_data = trainData[1]

X = tf.placeholder(tf.float32, shape=[None, 13])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([13, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
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