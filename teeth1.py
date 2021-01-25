import numpy as np
import csv
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

class Config(object):
    learning_rate = 0.01
    num_batches = 1000
    batch_size = 32
    display_step = 10
    depth = 128
    strides = [5, 3, 3, 2]
    kernel_size = [10, 5, 5, 5]
    fc_nodes = [1024]
    image_shape = (512, 512, 3)

config = Config()


WHITENESS = ['', '0M1', '0M2', '0M3', '1M1', '2M1',
            '2L1.5', '1M2', '2R1.5', '3M1', '2M2',
            '3R1.5', '2R2.5', '3L1.5', '2L2.5', '4M1',
            '2M3', '3M2', '4R1.5', '4L1.5', '3L2.5',
            '3R2.5', '5M1', '4M2', '3M3', '4R2.5',
            '4L2.5', '5M2', '4M3', '5M3'] 

if not os.path.exists('./checkpoints/'):
    os.mkdir('./checkpoints/')

target_file_path = './data/teeth_color.csv'

target_data = {}
with open(target_file_path, encoding='gbk') as f:
    csv_reader = csv.reader(f)
    _ = next(csv_reader)
    _ = next(csv_reader)
    for row in csv_reader:
        target_data.update({row[0]: [WHITENESS.index(i)-1 for i in row[1:]]})

# for name in target_data:
#     for i in target_data[name]:
#         if i and i not in WHITENESS:
#             print(name, i)

photo_path = './data/'
photo_data = {}
for name in target_data:
    if name in os.listdir(photo_path):
        photo_data.update({name: {}})
        for date in os.listdir(photo_path+'%s/' % name):
            day = os.listdir(photo_path+'%s/%s/' % (name, date))[0]
            if day == 'day0':
                photo_data[name].update({'baseline': []})
                for file_name in os.listdir(photo_path+'%s/%s/%s/baseline/' % (name, date, day)):
                    photo_data[name]['baseline'].append(photo_path+'%s/%s/%s/baseline/' % (name, date, day) + file_name)
                photo_data[name].update({'rau': []})
                for file_name in os.listdir(photo_path+'%s/%s/%s/首次使用后/' % (name, date, day)):
                    photo_data[name]['rau'].append(photo_path+'%s/%s/%s/首次使用后/' % (name, date, day) + file_name)
            else:
                photo_data[name].update({day: []})
                for file_name in os.listdir(photo_path+'%s/%s/%s/' % (name, date, day)):
                    photo_data[name][day].append(photo_path+'%s/%s/%s/' % (name, date, day) + file_name)

X = []
Y = []
for name in photo_data:
    for file_name in photo_data[name]['baseline']:
        X.append(file_name)
        Y.append(target_data[name][0])
    for file_name in photo_data[name]['rau']:
        X.append(file_name)
        Y.append(float(target_data[name][2]))
    n_days = len(photo_data[name].keys()) - 2
    for day in range(1, n_days + 1):
        for file_name in photo_data[name]['day%d' % day]:
            X.append(file_name)
            y0, y1 = target_data[name][0], target_data[name][4]
            Y.append( float((y1-y0)* day / n_days + y0) )   

X = np.array(X, dtype=str)

if os.path.exists('./data/PHOTOS.npy'):
    XX = np.load('./data/PHOTOS.npy')
else:
    XX = []
    print('Reading and Resizing Photos...')
    for i, x in enumerate(X):
        if not i%50:
            print(i, '/', len(X))
        xx = np.array(Image.open(x).resize(config.image_shape[:-1]))
        XX.append(xx)
    XX = np.array(XX)
    np.save('./data/PHOTOS.npy', XX)


Y = np.array(Y, dtype=float)
n_samples = X.shape[0]

def get_batch(x, y, batch_size=4):
    if 0 < batch_size < n_samples:
        choice = np.random.choice(np.arange(n_samples), size=batch_size, replace=False)
    else:
        choice = np.arange(n_samples)
    bx = x[choice].reshape((-1, ) + config.image_shape)
    by = y[choice].reshape((-1, 1))
    return bx, by

class CNN(object):
    def __init__(self):
        self.depth = config.depth
        self.n_layers = len(config.strides)
        self.strides = [[i, i] for i in config.strides]
        self.kernel_size = config.kernel_size
        self.fc_nodes = config.fc_nodes
        self.name = 'cnn_net'
        
    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h = x
            shapes = [h.get_shape().as_list()]
            for i_layer in range(self.n_layers):
                h = tf.layers.conv2d(inputs=h, filters=self.depth,
                 kernel_size=self.kernel_size[i_layer], strides=self.strides[i_layer], padding='valid')
                h = tf.nn.relu(h)
                # h = tf.layers.max_pooling2d(h, pool_size=2, strides=2)
                shapes.append(h.get_shape().as_list())
            h = tf.contrib.layers.flatten(h)
            shapes.append(h.get_shape().as_list())
            for nodes in self.fc_nodes:
                h = tf.contrib.layers.fully_connected(h, nodes, activation_fn=tf.nn.tanh)
            shapes.append(h.get_shape().as_list())
            h = tf.contrib.layers.fully_connected(h, 1)
            shapes.append(h.get_shape().as_list())
            return h, shapes
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

x = tf.placeholder(float, (None, ) + config.image_shape)
y_real = tf.placeholder(float, (None, 1))
cnn = CNN()
y_pred, shapes = cnn(x)
print(shapes)

loss = tf.reduce_mean(tf.square(y_pred - y_real))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
train_op = optimizer.minimize(loss)

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sess.run(tf.global_variables_initializer())

# tf.get_default_graph().finalize()


saver = tf.train.Saver()
bx, by = get_batch(XX, Y, batch_size=config.batch_size)
valid_feed = {x: bx, y_real: by}
info = []
for i_batch in range(config.num_batches):
    bx, by = get_batch(XX, Y, batch_size=config.batch_size)
    _ = sess.run([train_op, loss, y_pred], feed_dict={x: bx, y_real: by})
    if not (i_batch+1) % config.display_step:
        mse_loss, by_ = sess.run([loss, y_pred], feed_dict=valid_feed)
        acc = [np.mean(np.abs(by - by_) < 0.5), np.mean(np.abs(by - by_) < 1), np.mean(np.abs(by - by_) < 3)]
        info.append([mse_loss] + acc) 
        print('Iter %3d, mse loss: %6.4f' % (i_batch+1, np.sqrt(mse_loss)))
        print(acc)
