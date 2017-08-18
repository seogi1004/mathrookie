import numpy as np
import pandas as pd
import tensorflow as tf
import datetime

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

X = tf.placeholder(tf.float32, shape=[None, 13])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([13, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-10)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(50001):
    cost_val, hy_val, _ = sess.run([ cost, hypothesis, train ], feed_dict={ X: x_data, Y: y_data })
    if(step % 500 == 0):
        print(step, ", Cost: ", cost_val, "\nPrediction:\n", hy_val)