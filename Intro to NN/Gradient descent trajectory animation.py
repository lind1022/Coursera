'''
Introduction to Tensorflow
'''



import tensorflow.compat.v1 as tf
import numpy as np
import sys
import os
# from keras_utils import reset_tf_session
# s = reset_tf_session()
print("We're using TF", tf.__version__)
tf.disable_eager_execution()

DATA_FOLDER = '/Users/lin/github/Coursera/Intro to NN'
os.chdir(DATA_FOLDER)

# Implement a function computes the sum of squares of numbers from 0 to N-1
def sum_python(N):
    return np.sum(np.arange(N)**2)

sum_python(10**5)

# Doing the same thing with Tensorflow
s = tf.InteractiveSession()
# An integer parameter
N = tf.placeholder('int64', name="input_to_your_function")

# A recipe on how to produce the same result
result = tf.reduce_sum(tf.range(N)**2)

# Executing
s.run(result, {N: 10**5})

# logger for tensorboard
writer = tf.summary.FileWriter("tensorboard_logs", graph=s.graph)


'''
1. Define placeholders where you'll send inputs
2. Make a symbolic graph: a recipe for mathematical transformation of those placeholders
3. Compute outputs of your graph with particular values for each placeholder
    a) output.eval({placeholder: value})
    b) s.run(output, {placeholder: value})

So far there are two main entities: "placeholder" and "transformation" (operation output)

Both can be numbers, vectors, matrices, tensors, etc.
Both can be int32/64, floats, booleans (uint8) of various size.

You can define new transformations as an arbitrary operation on placeholders and other transformations

tf.reduce_sum(tf.arange(N)**2) are 3 sequential transformations of placeholder N
There's a tensorflow symbolic version for every numpy function
a+b, a/b, a**b, ... behave just like in numpy
np.mean -> tf.reduce_mean
np.arange -> tf.range
np.cumsum -> tf.cumsum
If you can't find the operation you need, see the docs.
tf.contrib has many high-level features, may be worth a look.
'''

with tf.name_scope("Placeholders_examples"):
    # Default placeholder that can be arbitrary float32
    # scalar, vertor, matrix, etc.
    arbitrary_input = tf.placeholder('float32')

    # Input vector of arbitrary length
    input_vector = tf.placeholder('float32', shape=(None,))

    # Input vector that _must_ have 10 elements and integer type
    fixed_vector = tf.placeholder('int32', shape=(10,))

    # Matrix of arbitrary n_rows and 15 columns
    # (e.g. a minibatch of your data table)
    input_matrix = tf.placeholder('float32', shape=(None, 15))

    # Can generally use None whenever you don't need a specific shape
    input1 = tf.placeholder('float64', shape=(None, 100, None))
    input2 = tf.placeholder('int32', shape=(None, None, 3, 224, 224))

    # elementwise multiplication
    double_the_vector = input_vector*2

    # elementwise cosine
    elementwise_cosine = tf.cos(input_vector)

    # difference between squared vector and vector itself plus one
    vector_squares = input_vector**2 - input_vector + 1

my_vector =  tf.placeholder('float32', shape=(None,), name="VECTOR_1")
my_vector2 = tf.placeholder('float32', shape=(None,))
my_transformation = my_vector * my_vector2 / (tf.sin(my_vector) + 1)
print(my_transformation)

dummy = np.arange(5).astype('float32')
print(dummy)

# Execution (option 1)
my_transformation.eval({my_vector: dummy, my_vector2: dummy[::-1]})
# Execution (option 2)
s.run(my_transformation, {my_vector: dummy, my_vector2: dummy[::-1]})

writer.add_graph(my_transformation.graph)
writer.flush()

!tensorboard --logdir=./tensorboard_logs/

'''
Summary:
1. Tensorflow is based on computation graphs
2. A graph consists of placeholders and transformations
'''






# End of script
