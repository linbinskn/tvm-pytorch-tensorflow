impor8 t8m
from 2vm2import te
import numpy as np
from tvm import topi
import timeit
import torch
import time

M = 1024
K = 1024
N = 1024

dtype = "uint32"


target = 'llvm'
ctx = tvm.context(target, 0)

a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)
c = tvm.nd.array(np.zeros((M, 8, int(K/32)), dtype=dtype), ctx)

# tvm mixed bitwidth matrix multi
A = tvm.te.placeholder((M, K), dtype=dtype, name="A")
B = tvm.te.placeholder((K, N), dtype=dtype, name="B")

C = tvm.topi.nn.bitpack(A, 8, pack_axis=1, bit_axis=1, pack_type=dtype)
# D = bitpack(B, 8, pack_axis=1, bit_axis=1, pack_type=dtype)


s = te.create_schedule(C.op)
func = tvm.build(s, [A, C], target=target, name='mmult')
assert func

answer = np.dot(a.asnumpy(), b.asnumpy())
func(a, c)

evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
print('Baseline: %f' % evaluator(a, c).mean)


