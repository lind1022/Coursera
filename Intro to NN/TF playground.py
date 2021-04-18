import tensorflow.compat.v1 as tf
import numpy as np

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
