from keras.layers import Input, Reshape, Dense, BatchNormalization, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model


def build_generator(random_noise_dimension, channels):
    # Generator attempts to fool discriminator by generating new images.
    model = Sequential()

    model.add(Dense(256 * 4 * 4,
                    activation="relu",
                    input_dim=random_noise_dimension))
    model.add(Reshape((4, 4, 256)))

    # Four layers of upsampling, convolution, batch normalization
    # and activation.
    # 1. Upsampling: Input data is repeated. Default is (2,2).
    #    In that case a 4x4x256 array becomes an 8x8x256 array.
    # 2. Convolution: If you are not familiar, you should watch
    #    this video: https://www.youtube.com/watch?v=FTr3n7uBIuE
    # 3. Normalization normalizes outputs from convolution.
    # 4. Relu activation:  f(x) = max(0,x). If x < 0, then f(x) = 0.

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Last convolutional layer outputs as many featuremaps as channels
    # in the final image.
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    # model.add(Conv2D(channels, kernel_size=1, padding="same"))
    # tanh maps everything to a range between -1 and 1.
    model.add(Activation("tanh"))

    # show the summary of the model architecture
    model.summary()

    # Placeholder for the random noise input
    my_input = Input(shape=(random_noise_dimension,))
    # Model output
    generated_image = model(my_input)

    # Change the model type from Sequential to Model (functional API)
    # More at: https://keras.io/models/model/.
    return Model(my_input, generated_image)