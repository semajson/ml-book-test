import numpy as np
from keras.preprocessing import image
import tensorflow as tf

files = ["horse-1.jpg", "horse-2.jpg", "human-1.jpg", "human-2.jpg"]


def horse_or_human(model, test_image):
    img = tf.keras.utils.load_img(test_image, target_size=(300, 300))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes)
    print(classes[0])
    if classes[0] > 0.5:
        print(test_image + " is a human")
    else:
        print(test_image + " is a horse")

# from test_horse_human import files, horse_or_human