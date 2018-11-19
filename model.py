import tensorflow as tf

LEARNING_RATE = 0.001
MOMENTUM = 0.99


def _create_conv_layer_(name, inputs, filters, size=5, stride=1, padding='same'):
    layer_name = 'Conv' + str(name) + '-' + str(size) + 'x' + str(size) + 'x' + str(filters) + '-' + str(stride)

    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=[size, size],
                            strides=[stride, stride],
                            activation=tf.nn.relu,
                            padding=padding,
                            name=layer_name)


def _create_deconv_layer_(name, inputs, filters, size=5, stride=1, padding='same'):
    layer_name = 'DeConv' + str(name) + '-' + str(size) + 'x' + str(size) + 'x' + str(filters) + '-' + str(stride)

    return tf.layers.conv2d_transpose(inputs=inputs,
                                      filters=filters,
                                      kernel_size=[size, size],
                                      strides=[stride, stride],
                                      activation=tf.nn.relu,
                                      padding=padding,
                                      name=layer_name)


def _create_pooling_layer_(name, inputs, size=2, stride=2, padding='same'):
    layer_name = 'Pool' + str(name) + '-' + str(size) + 'x' + str(size) + '-' + str(stride)
    return tf.layers.max_pooling2d(inputs=inputs,
                                   pool_size=[size, size],
                                   strides=[stride, stride],
                                   name=layer_name,
                                   padding=padding)


def _create_dense_layer_(name, inputs, nodes, activation=tf.nn.relu):
    layer_name = 'Dense' + str(name) + '-' + str(nodes)

    return tf.layers.dense(inputs=inputs,
                           units=nodes,
                           activation=activation,
                           name=layer_name)


def _get_loss_(prediction, truth):
    ignore_void_mask = tf.where(condition=tf.greater_equal(x=truth, y=0),
                                name='IgnoreVoid')

    non_void_truth = tf.gather_nd(params=truth, indices=ignore_void_mask, name='NonVoidTruth')
    non_void_prediction = tf.gather_nd(params=prediction, indices=ignore_void_mask, name='NonVoidPrediction')

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=non_void_truth,
                                           logits=non_void_prediction)
    return loss


def build_model(image_batch, true_segmentation):
    # Block 1
    image_batch = _create_conv_layer_(name='1', inputs=image_batch, size=3, filters=64)
    image_batch = _create_conv_layer_(name='2', inputs=image_batch, size=3, filters=64)
    image_batch = _create_pooling_layer_(name='1', inputs=image_batch, size=2, stride=2)

    # Block 2
    image_batch = _create_conv_layer_(name='3', inputs=image_batch, size=3, filters=128)
    image_batch = _create_conv_layer_(name='4', inputs=image_batch, size=3, filters=128)
    image_batch = _create_pooling_layer_(name='2', inputs=image_batch, size=2, stride=2)

    # Block 3
    image_batch = _create_conv_layer_(name='5', inputs=image_batch, size=3, filters=256)
    image_batch = _create_conv_layer_(name='6', inputs=image_batch, size=3, filters=256)
    image_batch = _create_conv_layer_(name='7', inputs=image_batch, size=3, filters=256)
    image_batch = _create_pooling_layer_(name='3', inputs=image_batch, size=2, stride=2)

    # Block 4
    image_batch = _create_conv_layer_(name='8', inputs=image_batch, size=3, filters=512)
    image_batch = _create_conv_layer_(name='9', inputs=image_batch, size=3, filters=512)
    image_batch = _create_conv_layer_(name='10', inputs=image_batch, size=3, filters=512)
    image_batch = _create_pooling_layer_(name='4', inputs=image_batch, size=2, stride=2)

    # Block 5
    image_batch = _create_conv_layer_(name='11', inputs=image_batch, size=3, filters=512)
    image_batch = _create_conv_layer_(name='12', inputs=image_batch, size=3, filters=512)
    image_batch = _create_conv_layer_(name='13', inputs=image_batch, size=3, filters=512)
    image_batch = _create_pooling_layer_(name='5', inputs=image_batch, size=2, stride=2)

    # Replaced FC
    image_batch = _create_conv_layer_(name='FC1', inputs=image_batch, size=7, filters=4096)
    image_batch = _create_conv_layer_(name='FC2', inputs=image_batch, size=1, filters=4096)
    image_batch = _create_conv_layer_(name='FC3', inputs=image_batch, size=1, filters=1)

    # Deconvolution Layer
    output = _create_deconv_layer_(name='DeConv1', inputs=image_batch, size=64, stride=32, filters=1)
    output = tf.nn.softmax(output, name='Softmax')

    loss = _get_loss_(prediction=output, truth=true_segmentation)
    optimize = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=MOMENTUM).minimize(loss=loss)

    return output, optimize, loss
