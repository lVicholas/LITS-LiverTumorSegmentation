import typing
import tensorflow as tf

def get_mobile_net_v3_encoder(
    model_input_shape: typing.Tuple[int] = (224, 224, 3),
    kernel_init_mu: float = 0.0,
    kernel_init_sigma: float = 0.02
):

    kernel_init_args = {'mean': kernel_init_mu, 'stddev': kernel_init_sigma}

    mobile_net_v3 = tf.keras.applications.MobileNetV3Large(
        input_shape=model_input_shape, 
        include_top=False,
        include_preprocessing=False,
    )

    encoder_output_layer_names = [
        're_lu_2',
        're_lu_6',
        're_lu_13',
        're_lu_22',
        're_lu_38'
    ]

    encoder_output_layers = [
        mobile_net_v3.get_layer(enc_out_layer_name).output 
        for enc_out_layer_name in encoder_output_layer_names
    ]

    hw, f = encoder_output_layers[-1].shape[-2:]
    final_relu = mobile_net_v3.get_layer(encoder_output_layer_names[-1])
    
    encoder_final_layer = tf.keras.layers.Conv2D(
        2*f, 
        3, 
        hw, 
        padding='same', 
        activation='relu',
        kernel_initializer=tf.random_normal_initializer(**kernel_init_args),
        name='final_encoding'
    )
    encoder_final_enc = encoder_final_layer(final_relu.output)

    encoder = tf.keras.Model(
        mobile_net_v3.input,
        encoder_output_layers + [encoder_final_enc,],
        name='encoder',
        trainable=False,
    )

    return encoder

def get_upsampler(
    n_filters: int, 
    size: int, 
    stride: int, 
    name: str, 
    dropout: float = 0, 
    init_sigma: float = 0.02
):

    kernel_init = tf.random_normal_initializer(0, init_sigma)

    conv_2d_t = tf.keras.layers.Conv2DTranspose(
        n_filters, 
        size,
        stride,
        padding='same', 
        kernel_initializer=kernel_init,
        name=f'conv_2d/{name}'
    )

    batch_norm = tf.keras.layers.BatchNormalization(name=f'batch_norm/{name}')
    activ = tf.keras.layers.ReLU(name=f'relu/{name}')

    upsampler = tf.keras.Sequential(name=name)
    upsampler.add(conv_2d_t)
    upsampler.add(batch_norm)

    if dropout != 0:
        dropout = tf.keras.layers.Dropout(dropout, name=f'dropout/{name}')
        upsampler.add(dropout)

    upsampler.add(activ)
    return upsampler

def get_decoder(encoder_ouput_num_filters: typing.List[int]):

    return [
        get_upsampler(f, 3, 2 if n != 0 else 7, f'up_{f}')
        for n, f in enumerate(reversed(encoder_ouput_num_filters[:-1]))
    ]

def get_liver_segmentation_unet(
    n_classes: int, 
    model_input_shape: typing.Tuple[int]
):

    encoder = get_mobile_net_v3_encoder(model_input_shape)
    encoder_ouput_num_filters =  [out[-1] for out in encoder.output_shape]
    decoder = get_decoder(encoder_ouput_num_filters)

    x_inp = tf.keras.layers.Input(model_input_shape, name='inp_x')
    encodings = encoder(x_inp)
    x = encodings[-1]
    skips = list(reversed(encodings[:-1]))

    for i, (up, sk) in enumerate(zip(decoder, skips)):

        x = up(x)
        x = tf.keras.layers.Concatenate(axis=-1)([x, sk])

        if i+1 < len(skips):

            h_x, w_x = x.shape[-3:-1]
            h_sk, w_sk, f_sk = sk.shape[-3:]
            stride_n = (h_sk // h_x, w_sk // w_x)

            new_layer = tf.keras.layers.Conv2DTranspose(
                f_sk, 
                3, 
                stride_n, 
                padding='same', 
                use_bias=False,
                name=f'conv2DT/concat_{i}'
            )

            x = new_layer(x)

    out = tf.keras.layers.Conv2DTranspose(
        n_classes, 3, 2, padding='same', name='output'
    )    

    x = out(x)

    return tf.keras.Model(x_inp, x, name='u_net')
