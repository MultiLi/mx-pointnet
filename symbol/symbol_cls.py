import mxnet as mx
from .symbol_common import get_fc_unit, data_transform


def get_pointnet_cls(dropout, num_class, reg_weight, is_training=True):
    data = mx.sym.var('data')

    # Rigid Transformation on input data
    input_transform = data_transform(data=data, K=3, prefix='input')
    data = mx.sym.batch_dot(lhs=data, rhs=input_transform, name='input_transform')

    # Point-wise Feature Extraction
    data = get_fc_unit(data=data, num_hidden=64, flatten=False, name='feature_fc1')
    data = get_fc_unit(data=data, num_hidden=64, flatten=False, name='feature_fc2')

    # Rigid Transformation on extracted feature
    feature_transform = data_transform(data=data, K=64, prefix='feature')
    data = mx.sym.batch_dot(lhs=data, rhs=feature_transform, name='feature_transform')

    # Global Feature Extraction
    data = get_fc_unit(data=data, num_hidden=64, flatten=False, name='global_feature_fc1')
    data = get_fc_unit(data=data, num_hidden=128, flatten=False, name='global_feature_fc2')
    data = get_fc_unit(data=data, num_hidden=1024, flatten=False, name='global_feature_fc3')
    data = mx.sym.max(data=data, axis=1, name='feature_pooling')

    # Classification
    data = get_fc_unit(data=data, num_hidden=512, flatten=False, name='cls_fc1')
    data = get_fc_unit(data=data, num_hidden=256, flatten=False, name='cls_fc2')
    data = mx.sym.Dropout(data=data, p=dropout, name='cls_dropout')
    data = mx.sym.FullyConnected(data=data, num_hidden=num_class, name='cls_fc3')

    data_group = data

    if is_training:
        data = mx.sym.SoftmaxOutput(data=data, normalization='batch', name='cls')
        transform_prod = mx.sym.batch_dot(lhs=feature_transform, rhs=feature_transform.swapaxes(1, 2))
        eye = mx.sym.one_hot(mx.sym.arange(start=0, stop=64), 64, name='eye')
        reg_loss = mx.sym.MakeLoss(
            data=reg_weight / 2 * mx.sym.broadcast_sub(lhs=transform_prod, rhs=eye) ** 2,
            name='reg_loss'
        )

        data_group = mx.sym.Group([data, reg_loss])

    return data_group


if __name__ == '__main__':
    cls_sym = get_pointnet_cls(dropout=0.3, num_class=10, reg_weight=0.001)
    net = mx.viz.plot_network(cls_sym)
    net.render()
