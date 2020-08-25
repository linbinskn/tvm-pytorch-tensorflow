import tvm
from tvm import te
import numpy as np
from tvm import topi
import timeit
import torch
import time
import tensorflow as tf

M = 1024
K = 1024
N = 1024

dtype = "uint32"


target = 'llvm'
ctx = tvm.context(target, 0)

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
C = topi.nn.bitserial_dense(A, B, 2, 2)

s = te.create_schedule(C.op)
func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

answer = np.dot(a.asnumpy(), b.asnumpy())
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
print('Baseline: %f' % evaluator(a, b, c).mean)


# pytorch matrix multi
sum_time = 0
for i in range(0,20):
	begin = time.time()
	mat1 = torch.randn(M, K)
	mat2 = torch.randn(K, N)
	torch.mm(mat1, mat2)
	end = time.time()
	sum_time += end-begin
sum_time = sum_time/20
print('Pytorch: %f' % sum_time)


# tensorflow matrix multi
a = np.random.randint(0, 2, (M, K))
b = np.random.randint(0, 2, (M, K))
mata = tf.convert_to_tensor(a)
matb = tf.convert_to_tensor(b)
product = tf.matmul(mata, matb)
with tf.Session() as sess:
	tf_time = 0
	for i in range(0, 20):	
		begin = time.time()
		result = sess.run([product])
		end = time.time()
		tf_time += end-begin
tf_time = tf_time/20
print('Tensorflow: %f' % tf_time)
