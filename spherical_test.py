import tensorflow as tf

from spherical import cubic_to_equirectangular
from spherical import equirectangular_to_cubic
from spherical import face_map

def read_image(image_path, shape):
    image = tf.image.decode_jpeg(tf.read_file(image_path))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, shape, tf.image.ResizeMethod.AREA)
    return image

def equirectangular_to_cubic_test():
    # Load equirectangular image.
    filename = "equirectangular"
    equirectangular_image = tf.expand_dims(read_image(filename + ".jpg", [256, 512]), 0)

    # Extract cube faces.
    cubic_images = equirectangular_to_cubic(equirectangular_image, [128, 128])
    session = tf.Session()

    # Save faces to disk.
    for index in range(6):
        cubic_image = cubic_images[index]
        face = face_map[index]
        quantized_image = tf.image.convert_image_dtype(cubic_image[0, :, :, :], tf.uint8)
        image_data = session.run(tf.image.encode_jpeg(quantized_image))
        with open("cubic_" + face + ".jpg", "w") as image_file:
            image_file.write(image_data)

def cubic_to_equirectangular_test():
    # Load cube faces.
    filenames = ["cubic_" + face + ".jpg" for face in face_map]
    cubic_images = [tf.expand_dims(read_image(filename, [128, 128]), 0) for filename in filenames]

    # Convert to equirectangular format.
    equirectangular_image = cubic_to_equirectangular(cubic_images, [256, 512])
    session = tf.Session()

    # Save to disk.
    quantized_image = tf.image.convert_image_dtype(equirectangular_image[0, :, :, :], tf.uint8)
    image_data = session.run(tf.image.encode_jpeg(quantized_image))
    with open("equirectangular_test.jpg", "w") as image_file:
        image_file.write(image_data)

if __name__ == "__main__":
    equirectangular_to_cubic_test()
    cubic_to_equirectangular_test()