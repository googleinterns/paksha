from jax import numpy, grad, xla_computation
from jax.lib import xla_client

'''
This file is an example of how to get optimized HLO from a JAX program.

For an example of how to get unoptimized HLO, see:
http://www.bnikolic.co.uk/blog/python/jax/2020/10/20/jax-outputgraph.html
'''

def tanh(x):  
  y = numpy.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

def lfn(x):
    return numpy.log(tanh(x).sum())

def dlfn(x):
    return  grad(lfn)(x)

xla_extension = xla_client._xla
xla_cpu_client = xla_client.get_local_backend("cpu")

comp = xla_computation(dlfn)(1.)
compiled_hlo = xla_cpu_client.compile(comp, xla_extension.CompileOptions())
module_compiled, = compiled_hlo.hlo_modules()

with open("optimized_hlo.txt", "w") as f:
    f.write(module_compiled.to_string())
    
