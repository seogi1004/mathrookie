import numpy as np
import pandas as pd
import datetime

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]

def createMatrixData(fileName, nb_classes):
    data = pd.read_csv(fileName)
    count = len(data["시간"])

    hours = []
    minutes = []
    outputs = []

    for i in range(count):
        dateObj = datetime.datetime.strptime(data["시간"][i], "%m/%d/%Y %H:%M")
        hours.append(dateObj.hour)
        minutes.append(dateObj.minute)

        cpu = data["프로세스 CPU사용률 (%)"][i]
        level = int(round(cpu))
        if(level >= nb_classes):
            level = nb_classes - 1

        outputs.append(level)

    data["구간"] = pd.Series(outputs, index=data.index)
    data["시"] = pd.Series(hours, index=data.index)
    data["분"] = pd.Series(minutes, index=data.index)

    cols = data.columns.tolist()
    xcols = cols[-2:] + cols[1:-4]
    ycols = cols[-3]

    print(xcols)
    print(ycols)

    xdata = data[xcols]
    ydata = data[ycols]

    # normalization
    xdata = (xdata - xdata.min()) / (xdata.max() - xdata.min())

    # standardization
    # xdata = (xdata - xdata.mean()) / xdata.std()

    x_data = xdata.as_matrix()
    y_data = ydata.as_matrix()
    one_hot_data = []

    for value in y_data:
        one_hot_data.append(
            one_hot_encode([ value ], nb_classes)[0]
        )

    return x_data, np.array(one_hot_data)

def createOriginalMatrixData(fileName):
    data = pd.read_csv(fileName)
    count = len(data["시간"])

    hours = []
    minutes = []

    for i in range(count):
        dateObj = datetime.datetime.strptime(data["시간"][i], "%m/%d/%Y %H:%M")
        hours.append(dateObj.hour)
        minutes.append(dateObj.minute)

    data["시"] = pd.Series(hours, index=data.index)
    data["분"] = pd.Series(minutes, index=data.index)

    cols = data.columns.tolist()
    xcols = cols[-2:] + cols[1:-3]
    ycols = cols[-3]

    xdata = data[xcols]
    ydata = data[ycols]

    # normalization
    xdata = (xdata - xdata.min()) / (xdata.max() - xdata.min())

    # standardization
    # xdata = (xdata - xdata.mean()) / xdata.std()

    x_data = xdata.as_matrix()
    y_data = ydata.as_matrix()

    return x_data, y_data

def createMergedMatrixData(todayName, yesterdayName):
    today_data = createOriginalMatrixData(todayName)
    yesterday_data = createOriginalMatrixData(yesterdayName)
    count = len(today_data[0])

    for i in range(count):
        today_x = today_data[0]
        today_y = today_data[1]
        yesterday_x = yesterday_data[0]
        yesterday_y = yesterday_data[1]

        if(today_x[i][2] == 0 and today_x[i][3] == 0 and today_x[i][4] == 0 and today_x[i][5] == 0):
            today_x[i][2] = yesterday_x[i][2]
            today_x[i][3] = yesterday_x[i][3]
            today_x[i][4] = yesterday_x[i][4]
            today_x[i][5] = yesterday_x[i][5]
            # today_y[i] = yesterday_y[i]

    return today_data[0], today_data[1]