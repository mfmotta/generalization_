import numpy as np
import keras

labels = np.asarray([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(float)


def norm(v):
    return (v-min(v))/(max(v)-min(v))

def layer_norm(layer):
    return np.linalg.norm(layer.get_weights()[0], 'fro')


def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test

def generate_data(x_train, y_train, size, labels=labels, random_labels=False, seed=False):
    if seed != False:
        np.random.seed(seed)
    ids = np.random.randint(x_train.shape[0], size=size)
    # print(ids)
    if random_labels == True:
        ys = y_train.copy()
        lb = labels.copy()
        np.random.shuffle(ys)
        np.random.shuffle(lb)
        y0 = ys[ids]
        y = np.asarray(
            [y0[i] if np.sum(y0[i] != y_train[ids][i]) else [yi for yi in labels if np.sum(yi != y0[i])][0] for i in
             range(size)])
        # print('assert',np.mean([np.sum((y==y_train[ids])[i,:])== 10 for i in range(size)]))
        return x_train[ids, :], y
    else:
        return x_train[ids, :], y_train[ids]


class MyEarlyStopping(keras.callbacks.Callback):

    def __init__(self, monitor='loss', monitor_value=0.0, period=0):
        super(MyEarlyStopping, self).__init__()
        self.monitor = monitor
        self.monitor_value = monitor_value
        self.period = period
        self.err = monitor_value / 10

    def on_train_begin(self, logs={}):
        self.epochs = []

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        self.period += 1

        if current is None:
            warnings.warn('Can save model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)


        elif logs.get(self.monitor) <= self.monitor_value + self.err and logs.get(
                self.monitor) >= self.monitor_value - self.err:

            self.epochs.append(self.period)
            self.model.stop_training = True





