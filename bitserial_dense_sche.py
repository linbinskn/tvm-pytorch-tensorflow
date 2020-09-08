import tvm
from tvm import te
import numpy as np
from tvm import topi
import timeit
import torch
import time
import tensorflow as tf
from tvm import autotvm

import tvm.topi.testing
from tvm.topi.util import get_const_int, get_const_tuple
from tvm.topi.nn.bitserial_util import bitpack, binary_op_multiplier
from tvm.topi.nn import bitserial_dense
from tvm.topi.x86 import schedule_reduce

M = 1024
K = 1024
N = 1024

dtype = "uint32"

target = 'llvm -mcpu=core-avx2'
# target = 'llvm'
ctx = tvm.context(target, 0)

_bitserial_dense_implement = {
    "generic": (topi.nn.bitserial_dense, topi.generic.schedule_bitserial_dense),
    "cpu": (topi.x86.bitserial_dense, topi.x86.schedule_bitserial_dense),
    "arm_cpu": (topi.arm_cpu.bitserial_dense, topi.arm_cpu.schedule_bitserial_dense),
}

a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)
c = tvm.nd.array(np.zeros((M, N), dtype='int16'), ctx)


# numpy matrix multi
np_repeat = 100
np_runing_time = timeit.timeit(setup='import numpy\n'
                                     'M = ' + str(M) + '\n'
                                     'K = ' + str(K) + '\n'
                                     'N = ' + str(N) + '\n'
                                     'dtype = "float32"\n'
                                     'a = numpy.random.rand(M, K).astype(dtype)\n'
                                     'b = numpy.random.rand(K, N).astype(dtype)\n',
                               stmt='answer = numpy.dot(a, b)',
                               number=np_repeat)
print("Numpy running time: %f" % (np_runing_time / np_repeat))

# tvm mixed bitwidth matrix multi
A = tvm.te.placeholder((M, K), dtype=dtype, name="A")
B = tvm.te.placeholder((K, N), dtype=dtype, name="B")
# C = bitserial_dense(A, B, 2, 2, unipolar=False)
fcompute, fschedule = tvm.topi.testing.dispatch(target, _bitserial_dense_implement)


input_dtype='uint32'
out_dtype='int16'
unipolar=False
# C = fcompute(A, B, 4, 4, input_dtype, out_dtype, unipolar)
# s = fschedule([C])
C = tvm.topi.x86.bitserial_dense(A, B, 2, 2, input_dtype, out_dtype, unipolar)
s = tvm.topi.x86.schedule_bitserial_dense(C)
print(tvm.lower(s, [A, B, C], simple_mode=True))
func = tvm.build(s, [A, B, C], target)
func(a, b, c)
#tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-5)
answer = np.dot(a.asnumpy(), b.asnumpy())
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
print('Baseline: %f' % evaluator(a, b, c).mean)





# pytorch matrix multi
sum_time = 0
mat1 = torch.randn(M, K, dtype=dtype)
mat2 = torch.randn(K, N, dtype=dtype)
begin = time.time()
for i in range(0,500):
	torch.mm(mat1, mat2)
end = time.time()
print('Pytorch: %f' ,(end-begin)/500)


# tensorflow matrix multi
a = np.random.randint(0, 2, (M, K))
b = np.random.randint(0, 2, (M, K))
mata = tf.convert_to_tensor(a)
matb = tf.convert_to_tensor(b)
product = tf.matmul(mata, matb)
begin = time.time()
with tf.Session() as sess:
	tf_time = 0
	for i in range(0, 500):
		result = sess.run([product])
end = time.time()
tf_time = (end-begin)/500
print('Tensorflow: %f' % tf_time)
