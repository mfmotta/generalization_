"""
Train and validate with adam for at most 3000 epochs or until with batch size 128:
     python train.py --random_labels=False --opt=Adam --max_epcs=3000 --bs=128 --max_size=128 --size=128 --target_loss_value=0.00001 --seed=False --path='/home/mariele/PycharmProjects/generalization/'
"""


import os
from functions import*
import keras
from keras.models import Sequential
import keras.layers as ll
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('random_labels', True, 'True to use random labels, False for correct.')
flags.DEFINE_integer('max_epcs', 3000, 'max number of epochs.')
flags.DEFINE_integer('bs', 128, 'batch size.')
flags.DEFINE_integer('max_size', 1024, 'maximun X_train size')
flags.DEFINE_integer('size', 128, 'minimum X_train size, must be >= bs')
flags.DEFINE_float('target_loss_value', 0.00001, 'proxy for loss minimun')
flags.DEFINE_string('opt', 'adam', 'optimizer')
flags.DEFINE_bool('seed', True, 'False to training with random seed')
flags.DEFINE_integer('seed_value', 0, 'keep same seed value to keep training consistent')
flags.DEFINE_string('path', '/home/mariele/PycharmProjects/generalization/', 'path to project: output arrays will be saved in files folder')


assert FLAGS.bs<=FLAGS.size,"minimum X_train size must be >= batch size"

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
y_train,y_val,y_test = map(keras.utils.np_utils.to_categorical,[y_train,y_val,y_test])

model = Sequential(name="mlp")
model.add(ll.InputLayer([28, 28]))
model.add(ll.Flatten())
l1=ll.Dense(64,name='l1')
l2=ll.Dense(64,name='l2')
l3=ll.Dense(10,name='l3',activation='softmax')
model.add(l1)
model.add(ll.Activation('relu'))
model.add(l2)
model.add(ll.Activation('relu'))
model.add(l3)

model.compile(optimizer=FLAGS.opt,loss="categorical_crossentropy", metrics=["accuracy"])
model.save_weights('model_gen.h5')

class MyModelCheckpoint(keras.callbacks.Callback):
    """see https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L360
    """

    def __init__(self, monitor='acc', monitor_value=0.0, period=0):
        super(MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.monitor_value = monitor_value
        self.period = period

    def on_train_begin(self, logs={}):
        self.losses, self.vlosses = [], []
        self.accuracies, self.vaccuracies = [], []
        self.norms1, self.norms2, self.norms3 = [], [] ,[]
        self.epochs = []

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        self.period += 1

        if current is None:
            warnings.warn('Can save model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)

        elif logs.get(self.monitor) == self.monitor_value:
            self.losses.append(logs.get('loss'))
            self.accuracies.append(logs.get('acc'))
            self.vaccuracies.append(logs.get('val_acc'))
            self.vlosses.append(logs.get('val_loss'))
            self.norms1.append(np.linalg.norm(l1.get_weights()[0], 'fro'))
            self.norms2.append(np.linalg.norm(l2.get_weights()[0], 'fro'))
            self.norms3.append(np.linalg.norm(l3.get_weights()[0], 'fro'))
            self.epochs.append(self.period)

def train(x_train, y_train, x_val, y_val, total_size=FLAGS.max_size, batch_size=FLAGS.bs, labels=labels, size=50, prt_rt=100,
            random_labels=FLAGS.random_labels, seed=FLAGS.seed):

    if seed != False:
        np.random.seed(FLAGS.seed_value)

    l2_norms, l2_norms1, losses, losses1 = [], [], [], []
    accs, accs1,vaccs, vaccs1 = [], [], [], []
    sizes, epochs, epochs1 = [], [], []
    val_losses, val_losses1 = [], []
    updates, updates1 = [], []
    hs,h1s = [],[]

    X_t, y_t = generate_data(x_train, y_train, size=total_size, labels=labels, random_labels=FLAGS.random_labels, seed=seed)

    size_ = size
    i = int(X_t.shape[0] / size_) - 1
    n_epochs = FLAGS.max_epcs #train for at most max_epcs, or until either accuracy=1 or loss ->0

    while i >= 0 and i * size <= total_size and (i + 1) * size >= 16:

        sizes.append(y_t[:(i + 1) * size].shape[0])
        model.load_weights('model_gen.h5')

        cb = MyModelCheckpoint(monitor='acc', monitor_value=1.0)
        es = MyEarlyStopping(monitor='loss', monitor_value=FLAGS.target_loss_value)
        training = model.fit(X_t[:(i + 1) * size], y_t[:(i + 1) * size], batch_size, epochs=n_epochs, verbose=0,
                             validation_data=(x_val[:(i + 1) * int(size / 5)], y_val[:(i + 1) * int(size / 5)]),
                             callbacks=[cb, es]);

        if es.epochs == []:
            print('not enough epochs for target loss value')
        if cb.epochs != []:
            accs1.append(cb.accuracies[0])
            F11, F21, F31 = cb.norms1[0], cb.norms2[0], cb.norms3[0]
            l2_norms1.append(4 * F11 ** 2 * F21 ** 2 * F31 ** 2)
            epochs1.append(cb.epochs[0])
            updates1.append(cb.epochs[0] * y_t[:(i + 1) * size].shape[0] / batch_size)
            losses1.append(cb.losses[0])
            val_losses1.append(cb.vlosses[0])
            vaccs1.append(cb.vaccuracies[0])

        F1, F2, F3 = layer_norm(l1), layer_norm(l2),layer_norm(l3)  # print(F11[-1],F1) checks!
        l2_norms.append(4 * F1 ** 2 * F2 ** 2 * F3 ** 2)
        losses.append(training.history['loss'][-1])
        val_losses.append(training.history['val_loss'][-1])
        accs.append(training.history['acc'][-1])
        vaccs.append(training.history['val_acc'][-1])
        epochs.append(cb.epochs[-1])  # equivalent to es.epochs[0]
        updates.append(epochs[-1] * y_t[:(i + 1) * size].shape[0] / batch_size)

        if (i + 1) * size % prt_rt == 0:
            print('size:', y_t[:(i + 1) * size].shape[0], '  --  loss:', np.round(training.history['loss'][-1], 6),
                  'after', epochs[-1], 'epochs')
        i = i - 1

    a_, va_, l_, vl_, l2_, e_, u_ = np.asarray(accs), np.asarray(vaccs), np.asarray(losses), np.asarray(
        val_losses), np.asarray(l2_norms), np.asarray(epochs), np.asarray(updates)
    a_1, va_1, l_1, vl_1, l2_1, e_1, u_1 = np.asarray(accs1), np.asarray(vaccs1), np.asarray(losses1), np.asarray(
        val_losses1), np.asarray(l2_norms1), np.asarray(epochs1), np.asarray(updates1)

    return np.asarray(sizes), a_, va_, l_, vl_,l2_, e_, u_, a_1, va_1, l_1, vl_1, l2_1,  e_1, u_1

tr = train(X_train, y_train, X_val, y_val, total_size = FLAGS.max_size, size = FLAGS.size,labels=labels, prt_rt = 32, random_labels = FLAGS.random_labels,
            seed = True)
sizes, a_, va_, l_, vl_, l2_, e_, u_, a_1, va_1, l_1, vl_1, l2_1,e_1, u_1 = tr

if FLAGS.random_labels==True:
    name = 'random'
else:
    name = 'correct'


np.savez(os.path.join(FLAGS.path+'files/loss0_arrays_'+name),sizes=sizes, acc_=a_, val_acc_=va_, loss_=l_, val_loss_=vl_, L2_=l2_,epcs_=e_, upts_=u_)
np.savez(os.path.join(FLAGS.path+'files/acc1_arrays_'+name),sizes=sizes, acc_1=a_1, val_acc_1=va_1, loss_1=l_1, val_loss_1=vl_1, L2_1=l2_1, epcs_1=e_1, upts_1=u_1)

