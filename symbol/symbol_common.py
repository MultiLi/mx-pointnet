import mxnet as mx


def get_fc_unit(data, num_hidden, flatten, name):
    data = mx.sym.FullyConnected(data=data, num_hidden=num_hidden, flatten=flatten, name=name)
    data = mx.sym.BatchNorm(data=data, name=name + '_bn')
    return mx.sym.relu(data=data, name=name + '_relu')


def data_transform(data, K, prefix=''):
    fc1 = get_fc_unit(data=data, num_hidden=64, flatten=False, name=prefix + '_transform_fc1')

    fc2 = get_fc_unit(data=fc1, num_hidden=128, flatten=False, name=prefix + '_transform_fc2')
    fc3 = get_fc_unit(data=fc2, num_hidden=1024, flatten=False, name=prefix + '_transform_fc3')

    max_pool = mx.sym.max(data=fc3, axis=1, name=prefix + '_transform_pooling')

    fc4 = get_fc_unit(data=max_pool, num_hidden=512, flatten=False, name=prefix + '_transform_fc4')
    fc5 = get_fc_unit(data=fc4, num_hidden=256, flatten=False, name=prefix + '_transform_fc5')

    fc_final = mx.sym.FullyConnected(data=fc5, num_hidden=K * K, flatten=False, name=prefix + '_transform_final')

    return mx.sym.reshape(data=fc_final, shape=(-1, K, K), name=prefix + '_reshape')
