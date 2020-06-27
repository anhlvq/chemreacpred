from numpy import array, hstack, math
from numpy.random import uniform


def create_2targetsdata(n):
    x1 = array([math.sin(i) * (i / 10) + uniform(-5, 5) for i in range(n)]).reshape(n, 1)
    x2 = array([math.cos(i) * (i / 10) + uniform(-9, 5) for i in range(n)]).reshape(n, 1)
    x3 = array([(i / 50) + uniform(-10, 10) for i in range(n)]).reshape(n, 1)

    y1 = [x1[i] + x2[i] + x3[i] + uniform(-1, 4) + 15 for i in range(n)]
    y2 = [x1[i] - x2[i] - x3[i] - uniform(-4, 2) - 10 for i in range(n)]
    X = hstack((x1, x2, x3))
    Y = hstack((y1, y2))
    return X, Y
