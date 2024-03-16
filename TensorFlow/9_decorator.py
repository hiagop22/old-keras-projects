import tensorflow as tf
import time
from datetime import datetime

from tensorflow.python import profiler

@tf.function
def function(x):
    a = tf.constant([[2.0],[3.0]])
    b = tf.constant(4.0)

    return a + b

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/func/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

tf.summary.trace_on(graph=True, profiler=True)

# Call only one tf.function when tracing
z = function(2)
with writer.as_default():
    tf.summary.trace_export(
        name='function_trace',
        step=0,
        profiler_outdir=logdir)

