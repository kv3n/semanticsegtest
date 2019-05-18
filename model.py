import tensorflow as tf


def _create_conv_layer_(name, filters, in_shape=None, size=5, stride=1, padding='same', activation='relu'):
    layer_name = 'Conv' + str(name) + '-' + str(size) + 'x' + str(size) + 'x' + str(filters) + '-' + str(stride)

    if in_shape is None:
        return tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=size,
                                      strides=stride,
                                      padding=padding,
                                      activation=activation,
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='glorot_uniform',
                                      name=layer_name)
    else:
        return tf.keras.layers.Conv2D(filters=filters,
                                      kernel_size=size,
                                      strides=stride,
                                      padding=padding,
                                      activation=activation,
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='glorot_uniform',
                                      input_shape=in_shape,
                                      name=layer_name)


def _create_deconv_layer_(name, filters, size=5, stride=1, padding='same'):
    layer_name = 'DeConv' + str(name) + '-' + str(size) + 'x' + str(size) + 'x' + str(filters) + '-' + str(stride)

    return tf.keras.layers.Conv2DTranspose(filters=filters,
                                           kernel_size=size,
                                           strides=stride,
                                           padding=padding,
                                           use_bias=True,
                                           kernel_initializer='glorot_uniform',
                                           bias_initializer='glorot_uniform',
                                           name=layer_name)


def _create_pooling_layer_(name, size=2, stride=2, padding='same'):
    layer_name = 'Pool' + str(name) + '-' + str(size) + 'x' + str(size) + '-' + str(stride)
    return tf.keras.layers.MaxPooling2D(pool_size=size,
                                        strides=stride,
                                        padding=padding,
                                        name=layer_name)


def mask_out_void(truth, prediction):
    non_void_pixels = tf.greater_equal(x=truth, y=0, name='NonVoidPixels')

    ignore_void_mask = tf.where(condition=non_void_pixels, name='NonVoidMask')

    non_void_truth = tf.gather_nd(params=truth, indices=ignore_void_mask,
                                  name='NonVoidTruth')
    non_void_prediction = tf.gather_nd(params=prediction, indices=ignore_void_mask,
                                       name='NonVoidPrediction')

    return non_void_truth, non_void_prediction


def _get_loss_(truth, prediction):
    non_void_truth, non_void_prediction = mask_out_void(truth, prediction)

    loss = tf.reduce_mean(tf.square(tf.sigmoid(non_void_prediction) - non_void_truth))

    return loss


def build_model(input_shape):
    output = _create_deconv_layer_(name='DeConv1', size=64, stride=32, filters=1)
    keras_model = tf.keras.Sequential([
        # Block 0
        _create_conv_layer_(name='0', in_shape=input_shape, size=1, filters=1, activation=None),

        # Block 1
        _create_conv_layer_(name='1', size=3, filters=64),
        _create_conv_layer_(name='2', size=3, filters=64),
        _create_pooling_layer_(name='1', size=2, stride=2),

        # Block 2
        _create_conv_layer_(name='3', size=3, filters=128),
        _create_conv_layer_(name='4', size=3, filters=128),
        _create_pooling_layer_(name='2', size=2, stride=2),

        # Block 3
        _create_conv_layer_(name='5', size=3, filters=256),
        _create_conv_layer_(name='6', size=3, filters=256),
        _create_conv_layer_(name='7', size=3, filters=256),
        _create_pooling_layer_(name='3', size=2, stride=2),

        # Block 4
        _create_conv_layer_(name='8', size=3, filters=512),
        _create_conv_layer_(name='9', size=3, filters=512),
        _create_conv_layer_(name='10', size=3, filters=512),
        _create_pooling_layer_(name='4', size=2, stride=2),

        # Block 5
        _create_conv_layer_(name='11', size=3, filters=512),
        _create_conv_layer_(name='12', size=3, filters=512),
        _create_conv_layer_(name='13', size=3, filters=512),
        _create_pooling_layer_(name='5', size=2, stride=2),

        # Replaced FC
        _create_conv_layer_(name='FC1', size=7, filters=4096),
        _create_conv_layer_(name='FC2', size=1, filters=4096),
        _create_conv_layer_(name='FC3', size=1, filters=1),

        # Deconvolution Layer
        output])

    keras_model.compile(optimizer='adam',
                        loss=_get_loss_,
                        metrics=['accuracy'])
    #keras_model.metrics += [output]

    keras_model.summary()

    return keras_model


def build_output_functor(model):
    # with a Sequential model
    get_output = tf.keras.backend.function([model.layers[0].input],
                                           [model.layers[-1].output])
    return get_output

#################
# TESTING
#################
"""
tf.enable_eager_execution()
prediction = np.array([[[[0.0], [0.4], [0.2]], [[-1.0], [-0.2], [-0.6]]]])
labels = np.array([[[[1.0], [1.0], [0.0]], [[-1.0], [-1.0], [0.0]]]])
loss = _get_loss_(prediction, labels)
"""