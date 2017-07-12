import tensorflow as tf

from spherical import equirectangular_to_cubic
from spherical import face_map

def read_image(image_path, shape):
    image = tf.image.decode_jpeg(tf.read_file(image_path))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, shape, tf.image.ResizeMethod.AREA)
    return image

def equirectangular_to_cubic_test():
    filename = "equirectangular"
    equirectangular_image = tf.expand_dims(read_image(filename + ".jpg", [1024, 2048]), 0)
    cubic_images = equirectangular_to_cubic(equirectangular_image, [512, 512])
    session = tf.Session()

    for index in range(6):
        cubic_image = cubic_images[index]
        face = face_map[index]
        quantized_image = tf.image.convert_image_dtype(cubic_image[0, :, :, :], tf.uint8)
        image_data = session.run(tf.image.encode_jpeg(quantized_image))
        with open("cubic_" + face + ".jpg", "w") as image_file:
            image_file.write(image_data)

equirectangular_to_cubic_test()