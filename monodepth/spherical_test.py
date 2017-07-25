import tensorflow as tf

from image_utils import encode_image
from image_utils import read_image
from image_utils import write_image

from spherical import cubic_to_equirectangular
from spherical import equirectangular_to_cubic
from spherical import face_map

def equirectangular_to_cubic_test():
    # Load equirectangular image.
    filename = "equirectangular"
    equirectangular_image = read_image(filename + ".jpg", [1024, 2048])

    # Extract cube faces.
    cubic_images = equirectangular_to_cubic(equirectangular_image, [512, 512])
    session = tf.Session()

    # Save faces to disk.
    for index in range(6):
        cubic_image = cubic_images[index]
        face = face_map[index]
        image_data = session.run(encode_image(cubic_image))
        write_image(image_data, "cubic_" + face + ".jpg")

def cubic_to_equirectangular_test():
    # Load cube faces.
    filenames = ["cubic_" + face + ".jpg" for face in face_map]
    cubic_images = [read_image(filename, [512, 512]) for filename in filenames]

    # Convert to equirectangular format.
    equirectangular_image = cubic_to_equirectangular(cubic_images, [1024, 2048])
    session = tf.Session()

    # Save to disk.
    image_data = session.run(encode_image(equirectangular_image))
    write_image(image_data, "equirectangular_test.jpg")

if __name__ == "__main__":
    equirectangular_to_cubic_test()
    cubic_to_equirectangular_test()