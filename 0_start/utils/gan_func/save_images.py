import matplotlib.pyplot as plt
import numpy as np


def save_images(epoch, random_noise_dimension, generator, target_dir,
                target_fn, start_epoch=0):
    # Save generated images for demonstration purposes using matplotlib.pyplot.
    rows, columns = 5, 5
    noise = np.random.normal(0, 1, (rows * columns, random_noise_dimension))
    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5

    figure, axis = plt.subplots(rows, columns)
    image_count = 0
    for row in range(rows):
        for column in range(columns):
            axis[row, column].imshow(generated_images[image_count, :] ) #, cmap='spring')
            axis[row, column].axis('off')
            image_count += 1
    figure.savefig(target_dir + target_fn + "_%4d.png" % (start_epoch + epoch))
    plt.close()
