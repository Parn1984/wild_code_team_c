from keras.layers import Input, Dropout, Dense, Flatten, BatchNormalization, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model


def build_discriminator(image_shape, alpha=0.2, dropout=0.25, momentum=0.8):
    '''

    Parameters
    ----------
    image_shape
    alpha: leaky relu alpha
    dropout: dropout for overfitting prevention
    momentum: for batch nomrmalization

    Returns
    -------

    '''
    # Discriminator attempts to classify real and generated images
    model = Sequential()

    model.add(Conv2D(32,
                     kernel_size=3,
                     strides=2,
                     input_shape=image_shape,
                     padding="same"))
    # Leaky relu is similar to usual relu. If x < 0 then f(x) = x * alpha,
    # otherwise f(x) = x.
    model.add(LeakyReLU(alpha=alpha))

    # Dropout blocks some connections randomly. This help the model
    # to generalize better. 0.25 means that every connection has a 25%
    # chance of being blocked.
    model.add(Dropout(dropout))
    model.add(Conv2D(64,
                     kernel_size=3,
                     strides=2,
                     padding="same"))
    # Zero padding adds additional rows and columns to the image.
    # Those rows and columns are made of zeros.
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Dropout(dropout))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Dropout(dropout))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Dropout(dropout))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=momentum))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Dropout(dropout))
    # Flatten layer flattens the output of the previous layer to a single dimension.
    model.add(Flatten())
    # Outputs a value between 0 and 1 that predicts whether image is real or generated. 0 = generated, 1 = real.
    model.add(Dense(1, activation='sigmoid'))

    #model.summary()

    input_image = Input(image_shape)

    # Model output given an image.
    validity = model(input_image)

    return Model(input_image, validity)
