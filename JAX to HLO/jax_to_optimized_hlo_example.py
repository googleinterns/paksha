from jax import numpy, grad
from jax.lib import xla_client
from jax_to_hlo import jax_to_hlo

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

xla_computation_hlo = jax_to_hlo(dlfn,[ ("x" , xla_client.Shape("f32[100]")) ])
compiled_hlo = xla_cpu_client.compile(xla_computation_hlo, xla_extension.CompileOptions())
module_compiled, = compiled_hlo.hlo_modules()

with open("optimized_hlo.txt", "w") as f:
    f.write(module_compiled.to_string())
    