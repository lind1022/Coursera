
# set up

import tensorflow.compat.v1 as tf
import numpy as np
import os

DATA_FOLDER = '/Users/lin/github/Coursera/Intro to NN'
os.chdir(DATA_FOLDER)

tf.disable_eager_execution()




x = tf.placeholder(tf.float32, (30, 10))
w = tf.Variable(tf.random.uniform((10, 20)), name='w')

z = x @ w
print(z)

# if ran tf.compat.v1.placeholder(tf.float32, (30, 10)) 3 times
tf.placeholder(tf.float32, (30, 10))
tf.placeholder(tf.float32, (30, 10))
tf.placeholder(tf.float32, (30, 10))

tf.get_default_graph().get_operations()

tf.reset_default_graph()


tf.reset_default_graph()
a = tf.constant(np.ones((2, 2), dtype=np.float32))
b = tf.Variable(tf.ones((2, 2)))
c = a @ b

s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())
s.run(c)

###############################
# The first neural network
###############################

tf.reset_default_graph()
x = tf.get_variable("x", shape=(), dtype=tf.float32)
f = x**2
# To get a synchronized output
f = tf.Print(f, [x, f], "x, f:")

optimizer = tf.train.GradientDescentOptimizer(0.1)
# step = optimizer.minimize(f, var_list=[x])
step = optimizer.minimize(f)

s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())

for i in range(10):
    # _, curr_x, curr_f = s.run([step, x, f])
    # print(curr_x, curr_f)
    s.run([step, x, f])


# Logging with TensorBoard


tf.summary.scalar('curr_x', x)
tf.summary.scalar('curr_f', f)
summaries = tf.summary.merge_all()

optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(f)

s = tf.InteractiveSession()
summary_writer = tf.summary.FileWriter("logs/2", s.graph)
s.run(tf.global_variables_initializer())

for i in range(10):
	_,curr_summaries = s.run([step, summaries])
	summary_writer.add_summary(curr_summaries, i)
	summary_writer.flush

# Then run this in command line
!tensorboard --logdir=./logs/



################################
# Solving a linear regression
################################

# first generate a model dataset
# we have 1000 points with 3 dimensions
N = 1000
D = 3
x = np.random.random((N, D))
w = np.random.random((D, 1))
y = x @ w + np.random.randn(N, 1) * 0.2

# firstly we need some placeholders, one for features one for target
tf.reset_default_graph()
features = tf.placeholder(tf.float32, shape=(None, D)) # features matrix
target = tf.placeholder(tf.float32, shape=(None, 1)) # a single collumn target matrix

weights = tf.get_variable("w", shape=(D, 1), dtype=tf.float32)
predictions = features @ weights

# Loss function - mean squared error defined as tensorflow operations
loss = tf.reduce_mean((target - predictions) ** 2)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(loss)
# To do gradient descent

s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())
for i in range(300):
    _, curr_loss, curr_weights = s.run([step, loss, weights], feed_dict={features: x, target: y})
    if i % 50 == 0:
        print(curr_loss)


# Ground truth weights
w
# Estimated weights
curr_weights

##############################
# Another linear regression
##############################

m = tf.get_variable("m", [], initializer=tf.constant_initializer(0.))
b = tf.get_variable("b", [], initializer=tf.constant_initializer(0.))
init = tf.global_variables_initializer()

## then set up the computations
input_placeholder = tf.placeholder(tf.float32)
output_placeholder = tf.placeholder(tf.float32)

x = input_placeholder
y = output_placeholder
y_guess = m * x + b

loss = tf.square(y - y_guess)

## finally, set up the optimizer and minimization node
optimizer = tf.train.GradientDescentOptimizer(1e-3)
train_op = optimizer.minimize(loss)

### start the session
sess = tf.Session()
sess.run(init)

### perform the training loop
import random

## set up problem
true_m = random.random()
true_b = random.random()

for update_i in range(100000):
  ## (1) get the input and output
  input_data = random.random()
  output_data = true_m * input_data + true_b

  ## (2), (3), and (4) all take place within a single call to sess.run()!
  # Every time we call sess.run(), we perform a step of gradient descent
  _loss, _ = sess.run([loss, train_op], feed_dict={input_placeholder: input_data, output_placeholder: output_data})
  if update_i % 10000 == 0:
      print(update_i, _loss)

### finally, print out the values we learned for our two variables
print("True parameters:     m=%.4f, b=%.4f" % (true_m, true_b))
print("Learned parameters:  m=%.4f, b=%.4f" % tuple(sess.run([m, b])))

two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
print_sum_node = tf.Print(sum_node, [two_node, three_node])
sess = tf.Session()
print(sess.run(print_sum_node))


'''
One important, somewhat-subtle point about tf.Print: printing is a side effect.
Like all other side effects, printing only occurs if the computation flows through
the tf.Print node. If the tf.Print node is not in the path of the computation,
nothing will print. In particular, even if the original node that your tf.Print
node is copying is on the computation path, the tf.Print node itself might not be.
Watch out for this issue! When it strikes (and it eventually will), it can be
incredibly frustrating if you aren’t specifically looking for it. As a general
rule, try to always create your tf.Print node immediately after creating the
node that it copies.
'''

two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node
### this new copy of two_node is not on the computation path, so nothing prints!
print_two_node = tf.Print(two_node, [two_node, three_node, sum_node])
sess = tf.Session()
print(sess.run(sum_node))








# end of script
