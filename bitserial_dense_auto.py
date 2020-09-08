import tvm
from tvm import te
import numpy as np
from tvm import topi
import timeit
import torch
import time
import tensorflow as tf
from tvm import autotvm
from tvm.topi import tag
import logging
import sys
from tvm.topi.util import get_const_int, get_const_tuple
from tvm.topi.nn.bitserial_util import bitpack, binary_op_multiplier

M = 1024
K = 1024
N = 1024

dtype = "uint32"
target = 'llvm -mcpu=core-avx2'
# target = 'llvm'
ctx = tvm.context(target, 0)

@autotvm.template("bitserial_den")
def bitserial_dense():
    """Bitserial dense implementation. TODO: Why are these separate

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]
    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim] or
        3-D with shape [out_dim, weight_bits, in_dim]
    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    M = 1024
    K = 1024
    N = 1024
    data = tvm.te.placeholder((M, K), dtype=dtype, name="A")
    weight = tvm.te.placeholder((K, N), dtype=dtype, name="B")
    data_bits = 2
    weight_bits = 2
    pack_dtype = 'uint32'
    out_dtype = 'int16'
    unipolar = False
    cfg = autotvm.get_config()
    data_packed = bitpack(data, data_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    if len(weight.shape) == 2:
        weight_packed = bitpack(weight, weight_bits, pack_axis=1, bit_axis=1, pack_type=pack_dtype)
    else:
        weight_packed = weight
    Y, DB, K = get_const_tuple(data_packed.shape)
    X, WB, _ = get_const_tuple(weight_packed.shape)
    ######## Search space
    x, y = cfg.axis(X), cfg.axis(Y)
    db, wb, k = cfg.reduce_axis(DB), cfg.reduce_axis(WB), cfg.reduce_axis(K)
    ko, ki = cfg.define_split('tile_k', k, num_outputs=2)
    yo, yi = cfg.define_split('tile_y', y, num_outputs=2)
    xo, xi = cfg.define_split('tile_x', x, num_outputs=2)

    cfg.define_reorder('reorder_0', [yo, xo, ko, yi, wb, db, ki, xi],
                       policy='candidate', candidate=[
                           [yo, xo, ko, yi, wb, db, ki, xi],
                           [yo, xo, yi, ko, wb, db, ki, xi]])

    cfg.define_annotate('ann_reduce', [db, wb], policy='try_unroll')
    cfg.define_annotate('ann_spatial', [yi, xi], policy='try_unroll_vec')

    ###### Compute rule
    VX = cfg['tile_x'].size[-1]

    wvshape = (X//VX, WB, VX, K)
    oshape = (Y, X)

    k = te.reduce_axis((0, K), name='k')
    db = te.reduce_axis((0, DB), name='db')
    wb = te.reduce_axis((0, WB), name='wb')

    # Tile data and weights
    weight_vec = te.compute(wvshape, lambda xo, wb, vx, k:
                            weight_packed[xo*VX+vx][wb][k], name='weight_vec')

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    matmul_unipolar = te.compute(oshape, lambda i, j: te.sum(
        (tvm.tir.popcount(weight_vec[idxdiv(j, VX), wb, idxmod(j, VX), k] & data_packed[i, db, k]) -
         tvm.tir.popcount(~weight_vec[idxdiv(j, VX), wb, idxmod(j, VX), k] & data_packed[i, db, k])
         ).astype(out_dtype)
        << (db+wb).astype(out_dtype), axis=[wb, db, k]), tag='bitserial_dense_unipolar')

    matmul = te.compute(oshape, lambda i, j: te.sum(
        tvm.tir.popcount(weight_vec[idxdiv(j, VX), wb, idxmod(j, VX), k] & data_packed[i, db, k]
                         ).astype(out_dtype)
        << (db+wb).astype(out_dtype), axis=[wb, db, k]), tag='bitserial_dense')

    # binary ops
    cfg.add_flop(2 * Y * X * K * binary_op_multiplier(pack_dtype))

    #if unipolar:
    #    return matmul_unipolar
    #return matmul
    if unipolar:
        outs = matmul_unipolar
    else:
        outs = matmul
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(cfg, s, data_vec, weight_vec, output):
        s[data_vec].parallel(s[data_vec].op.axis[0])
        s[weight_vec].parallel(s[weight_vec].op.axis[0])

        y, x = s[output].op.axis
        wb, db, k = s[output].op.reduce_axis

        yo, yi = cfg["tile_y"].apply(s, output, y)
        xo, xi = cfg["tile_x"].apply(s, output, x)
        ko, ki = cfg["tile_k"].apply(s, output, k)


        cfg["reorder_0"].apply(s, output, [yo, xo, ko, yi, wb, db, ki, xi])
        cfg["ann_reduce"].apply(s, output, [db, wb],
                                axis_lens=[get_const_int(db.dom.extent),
                                           get_const_int(wb.dom.extent)],
                                max_unroll=8,
                                cfg=cfg)
        cfg["ann_spatial"].apply(s, output, [yi, xi],
                                 axis_lens=[cfg['tile_y'].size[-1],
                                            cfg['tile_x'].size[-1]],
                                 max_unroll=8,
                                 cfg=cfg)
        s[output].vectorize(xi)
        s[output].parallel(yo)
        return s

    def traverse(op):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag) or 'elemwise' in op.tag:
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.te.ComputeOp):
                    traverse(tensor.op)

        elif op.tag == 'bitserial_dense' or 'bitserial_dense_unipolar':
            output = op.output(0)
            weight_vec = op.input_tensors[0]

            data_vec = op.input_tensors[1]
            data = data_vec.op.input_tensors[0]
            if "QuantizeInput" in data.op.name:
                data = data.op.input_tensors[0]
            _schedule(cfg, s, data_vec, weight_vec, output)
        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

    traverse(outs[0].op)
    if unipolar:
        C = matmul_unipolar
    else:
        C = matmul
    return s, [data, weight, C]


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

# C = bitserial_dense(cfg, A, B, 2, 2, 'uint32', 'int16', False)

# s = schedule_bitserial_dense(cfg, C)
task = autotvm.task.create("bitserial_den", args=(), target=target)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(n_trial=3000,
           early_stopping=800,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('matmul.log')])

# apply history best from log file
with autotvm.apply_history_best('matmul.log'):
    with tvm.target.create(target):
        s, args = bitserial_dense()
        func = tvm.build(s, args, target=target, name='mmult')


# func = tvm.build(s, [A, B, C], target=target, name='mmult')
assert func

answer = np.dot(a.asnumpy(), b.asnumpy())
func(a, b, c)
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
