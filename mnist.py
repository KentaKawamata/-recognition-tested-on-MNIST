import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

mnist = datasets.fetch_mldata('MNIST original', data_home = '.')

n = len(mnist.data) 
N = 10000
indices = np.random.permutation(range(n))[:N]
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]       #出力層 1-0f-K表現で出力
X_train, X_test, Y_train, Y_test, =  train_test_split(X, Y, train_size=0.8, random_state=0)

def lrelu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)

keep_prob = tf.placeholder(tf.float32) #ドロップアウトしない確率

n_in = len(X[0])
num_hidden_0 = 200
x = tf.placeholder(tf.float32, shape=[None, n_in])
W_0 = tf.Variable(tf.truncated_normal([n_in, num_hidden_0], stddev=0.01))
b_0 = tf.Variable(tf.zeros([num_hidden_0]))
h_0 = tf.nn.relu(tf.matmul(x, W_0) + b_0)
h_drop0 = tf.nn.dropout(h_0, keep_prob)

num_hidden_1 = 200
W_1 = tf.Variable(tf.truncated_normal([num_hidden_0, num_hidden_1], stddev=0.01))
b_1 = tf.Variable(tf.zeros([num_hidden_1]))
h_1 = tf.nn.relu(tf.matmul(h_drop0, W_1) + b_1)
h_drop1 = tf.nn.dropout(h_1, keep_prob)

num_hidden_2 = 200
W_2 = tf.Variable(tf.truncated_normal([num_hidden_1, num_hidden_2], stddev=0.01))
b_2 = tf.Variable(tf.zeros([num_hidden_2]))
h_2 = tf.nn.relu(tf.matmul(h_drop1, W_2) + b_2)
h_drop2 = tf.nn.dropout(h_2, keep_prob)

num_hidden_3 = 200
W_3 = tf.Variable(tf.truncated_normal([num_hidden_2, num_hidden_3], stddev=0.01))
b_3 = tf.Variable(tf.zeros([num_hidden_3]))
h_3 = tf.nn.relu(tf.matmul(h_drop2, W_3) + b_3)
h_drop3 = tf.nn.dropout(h_3, keep_prob)

n_out = 10
V = tf.Variable(tf.truncated_normal([num_hidden_3, n_out], stddev=0.01))
c = tf.Variable(tf.zeros([n_out]))
y = tf.nn.softmax(tf.matmul(h_drop3, V) + c)

t = tf.placeholder(tf.float32, shape=[None, n_out])
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1-t) * tf.log(1-y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 100
epochs = 1000

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(epochs):
    X_, Y_ = shuffle(X_train, Y_train)
   
    for i in range(batch_size):
        start = i * batch_size
        end = start + batch_size
        #[start:end]は、startオフセットからend-1オフセットまでのシーケンスを抽出する
        sess.run(train_step, feed_dict={x: X_[start:end], t: Y_[start:end], keep_prob: 0.5})
        

accuracy_rate = accuracy.eval(session=sess, feed_dict={x: X_test, t: Y_test, keep_prob: 1.0})
print('認識精度:', accuracy_rate)

