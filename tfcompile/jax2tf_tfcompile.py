import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
import tf2xla_pb2

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

def f(a, b):
  return jax.lax.add(a, b).sum()

f = tf.function(jax2tf.convert(f))
a = b = tf.ones([1, 1])
cf = f.get_concrete_function(a, b)

graph_def = cf.graph.as_graph_def()
with open('graph.pb', 'wb') as fp:
  fp.write(graph_def.SerializeToString())

config = tf2xla_pb2.Config()
batch_size = 1

feeds = [o.name for o in cf.graph.get_operations() if o.name.startswith('jax2tf_arg')]
fetches = [o.name for o in cf.graph.get_operations() if o.name.startswith('jax2tf_out')]

for idx, x in enumerate(cf.inputs):
	x.set_shape([batch_size] + list(x.shape)[1:])
	feed = config.feed.add()
	feed.id.node_name = feeds[idx]
	feed.shape.MergeFrom(x.shape.as_proto())

for f in fetches:
	fetch = config.fetch.add()
	fetch.id.node_name = f

with open('graph.config.pbtxt', 'w') as f:
	f.write(str(config))
