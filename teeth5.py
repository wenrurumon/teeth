import time
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import csv
import pandas as pd
import numpy as np
from PIL import Image


TRAINING = True

if TRAINING:
    TIME_STAMP = time.strftime('%m%d%H%M')
else:
    TIME_STAMP = '01110359'

MODEL_NAME = 'TEETH_CNN_MOUSE_%s' % TIME_STAMP
SAVE_PATH = './results/'


class Config(object):
    learning_rate = 0.01
    decay_rate = 0.001
    num_batches = 30000
    batch_size = 50
    display_step = 100
    depths = [32, 64, 128]
    strides = [5, 3, 3]
    kernel_size = [20, 10, 5]
    fc_nodes = [1024]
    keep_prob = 0.9
    image_shape = (128, 128, 6)
    reg_alpha = 0.1
    output_size = 4


config = Config()
np.random.seed(1234)


try:
    print('Loading Existing Photos Numpy File...')
    XX = np.load('./data/mouse_data.npy')
    YY = np.load('./data/mouse_labels.npy')
    G = np.load('./data/groups.npy')
except:
    map = pd.read_csv('./data/x2model.csv')
    # map.head
    n_maps = len(map['id'])
    print('Reading Photo Files...')
    ids = []
    mouse = []
    card = []
    labwio = []
    for i in range(n_maps):
        if not i % 100:
            print(i, '/', n_maps)
        if isinstance(map['id'][i], str):
            ids.append(map['id'][i].split('@'))
        else:
            continue
        mouse.append(np.array(Image.open(map['mouse'][i].replace(
            './', './data/')).resize(config.image_shape[:-1])))
        card.append(np.array(Image.open(map['card'][i].replace(
            './', './data/')).resize(config.image_shape[:-1])))
        labwio.append([map['l'][i], map['a'][i], map['b'][i], map['wio'][i]])

    names = sorted(list(set([i[0] for i in ids if i])))
    valid_names = [i for i in names if np.random.random() < 0.1]
    train_names = [i for i in names if i not in valid_names]

    G = [name[0] in valid_names for name in ids]  # 1 for validation sample, and 0 for training sample

    XX = np.array([np.concatenate([m, c], axis=-1) for m, c in zip(mouse, card)])
    YY = np.array(labwio)

    np.save('./data/mouse_data.npy', XX)
    np.save('./data/mouse_labels.npy', YY)
    np.save('./data/groups.npy', G)

n_samples = XX.shape[0]

# NUMPY_FILE_NAME = 'PHOTO_128m'
# WHITENESS = ['', '0M1', '0M2', '0M3', '1M1', '2M1',
#              '2L1.5', '1M2', '2R1.5', '3M1', '2M2',
#              '3R1.5', '2R2.5', '3L1.5', '2L2.5', '4M1',
#              '2M3', '3M2', '4R1.5', '4L1.5', '3L2.5',
#              '3R2.5', '5M1', '4M2', '3M3', '4R2.5',
#              '4L2.5', '5M2', '4M3', '5M3']
METRICS = [0.5, 1, 2, 3, 5]
# Normalize YY
Y_means = [np.mean(YY[:, i]) for i in range(config.output_size)]
Y_stds = [np.std(YY[:, i]) for i in range(config.output_size)]
for i in range(config.output_size):
    YY[:, i] = (YY[:, i] - Y_means[i]) / Y_stds[i]

if not os.path.exists('./checkpoints/'):
    os.mkdir('./checkpoints/')

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)


valid_ids = [i for i in range(n_samples) if G[i]]
train_ids = [i for i in range(n_samples) if not G[i]]

# valid_ids = [i for i in range(n_samples) if np.random.random() < 0.1]
# train_ids = [i for i in range(n_samples) if i not in valid_ids]


x = tf.placeholder(float, (None, ) + config.image_shape)
y_real = tf.placeholder(float, (None, config.output_size))
keep_prob = tf.placeholder(tf.float32, ())
learning_rate = tf.placeholder(tf.float32, ())
regularizer = tf.contrib.layers.l2_regularizer(scale=config.reg_alpha)


def get_batch(x, y, batch_size=10, mode='train'):
    x_shape = x.shape[1:]
    y_shape = y.shape[1:]
    if batch_size > 0:
        if mode == 'train':
            choice = np.random.choice(
                train_ids, size=batch_size, replace=False)
        elif mode == 'valid':
            choice = np.random.choice(
                valid_ids, size=batch_size, replace=False)
    else:
        choice = np.arange(n_samples)
    bx = x[choice].reshape((-1, ) + x_shape)
    by = y[choice].reshape((-1, ) + y_shape)
    return bx, by


class CNN(object):
    def __init__(self):
        self.depths = config.depths
        self.n_layers = len(config.strides)
        self.strides = [[i, i] for i in config.strides]
        self.kernel_size = config.kernel_size
        self.fc_nodes = config.fc_nodes
        self.output_size = config.output_size
        self.name = MODEL_NAME

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h = x
            shapes = [h.get_shape().as_list()]
            for i_layer in range(self.n_layers):
                h = tf.layers.conv2d(inputs=h, filters=self.depths[i_layer],
                                     kernel_size=self.kernel_size[i_layer], strides=self.strides[i_layer],
                                     kernel_regularizer=regularizer, padding='same')
                h = tf.nn.leaky_relu(h)
                # h = tf.layers.max_pooling2d(h, pool_size=2, strides=2, padding='same')
                shapes.append(h.get_shape().as_list())

            h = tf.contrib.layers.flatten(h)
            shapes.append(h.get_shape().as_list())
            for nodes in self.fc_nodes:
                h = tf.contrib.layers.fully_connected(h, nodes,
                                                      weights_regularizer=regularizer, activation_fn=tf.nn.leaky_relu)
                h = tf.nn.dropout(h, keep_prob=keep_prob)

            shapes.append(h.get_shape().as_list())
            h = tf.contrib.layers.fully_connected(
                h, self.output_size, activation_fn=None)
            shapes.append(h.get_shape().as_list())
            return h, shapes

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


cnn = CNN()
y_pred, shapes = cnn(x)
print(shapes)

# exit()

loss = tf.reduce_mean(tf.square(y_pred - y_real))
l2_loss = tf.losses.get_regularization_loss()
loss += l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sess.run(tf.global_variables_initializer())

# tf.get_default_graph().finalize()


saver = tf.train.Saver()
save_file = './checkpoints/%s.ckpt' % cnn.name

if TRAINING:
    vx, vy = get_batch(XX, YY, batch_size=config.batch_size, mode='valid')
    valid_feed = {x: vx, y_real: vy, keep_prob: 1.0}
    info = []
    for i_batch in range(config.num_batches):
        current_learning_rate = config.learning_rate * \
            config.decay_rate ** (i_batch/config.num_batches)
        bx, by = get_batch(XX, YY, batch_size=config.batch_size, mode='train')
        train_feed = {x: bx, y_real: by, keep_prob: config.keep_prob, learning_rate: current_learning_rate}
        _, loss_train, by_ = sess.run([train_op, loss, y_pred], feed_dict=train_feed)
        acc_train = [np.mean(np.sqrt(np.mean(np.square(by - by_))) < metric) for metric in METRICS]
        if not (i_batch+1) % config.display_step:
            loss_valid, vy_ = sess.run([loss, y_pred], feed_dict=valid_feed)
            acc_valid = [np.mean(np.sqrt(np.mean(np.square(vy - vy_))) < metric)
                         for metric in METRICS]
            info.append([loss_train, loss_valid] + acc_train + acc_valid)
            print('Iter %3d, train loss: %.3f, valid loss: %.3f, lr: %.2e' % (
                i_batch+1, np.sqrt(loss_train), np.sqrt(loss_valid), current_learning_rate))
            print('Train Acc:', acc_train)
            print('Valid Acc:', acc_valid)
            # print(np.round(error, 2).flatten())
            # print(np.round(by_, 2).flatten())

    saver.save(sess, save_file)
    info = np.array(info).T
    np.save(SAVE_PATH + 'info_%s.npy' % MODEL_NAME, info)
else:
    saver.restore(sess, save_file)
    info = np.load(SAVE_PATH + 'info_%s.npy' % MODEL_NAME)

plt.figure(dpi=200)
plt.plot(info[0])
plt.plot(info[1])
plt.legend(['train_loss', 'valid_loss'])
plt.xlabel('Number of Batches x%d' % config.display_step)
plt.savefig(SAVE_PATH + 'mse_loss_%s.png' % MODEL_NAME)
plt.close()

plt.figure(dpi=200)
for i in info[2:2+len(METRICS)]:
    plt.plot(i)
plt.xlabel('Number of Batches x%d' % config.display_step)
plt.legend(['%.1f acc' % metric for metric in METRICS])
plt.savefig(SAVE_PATH + 'acc_train_%s.png' % MODEL_NAME)
plt.close()

plt.figure(dpi=200)
for i in info[2+len(METRICS):]:
    plt.plot(i)
plt.xlabel('Number of Batches x%d' % config.display_step)
plt.legend(['%.1f acc' % metric for metric in METRICS])
plt.savefig(SAVE_PATH + 'acc_valid_%s.png' % MODEL_NAME)
plt.close()


# bx, by = get_batch(XX, Y, batch_size=30)
# by_ = sess.run(y_pred, feed_dict={x: bx, keep_prob: 1.0}).flatten()
# by = by.flatten()
# for i, j in zip(by, by_):
#     print(i, j)
# print(np.corrcoef(by, by_)[1, 0])

Y_ = np.zeros_like(YY)

for i in range(n_samples // config.batch_size + 1):
    choice = np.arange(i * config.batch_size,
                       np.min([n_samples, (i+1)*config.batch_size]))
    bx = XX[choice].reshape((-1, ) + config.image_shape)
    by_ = sess.run(y_pred, feed_dict={x: bx, keep_prob: 1.0})
    Y_[choice] = by_
    # Y_[choice] = i


with open(SAVE_PATH + 'Results_%s.csv' % MODEL_NAME, 'w+', newline='') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['Photo ID', 'Ground Truth',
                        'Predicted', 'Error', 'Group'])
    for i in range(n_samples):
        if i in train_ids:
            group = 'TRAIN'
        elif i in valid_ids:
            group = 'VALID'
        csvwriter.writerow([i+1, YY[i], Y_[i], np.sqrt(np.mean(np.square(YY[i]-Y_[i]))), group])

ACC_TRAIN = [np.round(np.mean(
    np.sqrt(np.mean(np.square(YY[train_ids] - Y_[train_ids]))) < metric), 4) for metric in METRICS]
ACC_VALID = [np.round(np.mean(
    np.sqrt(np.mean(np.square(YY[valid_ids] - Y_[valid_ids]))) < metric), 4) for metric in METRICS]

print('Train accuracy:', ACC_TRAIN)
print('Validation accuracy:', ACC_VALID)
