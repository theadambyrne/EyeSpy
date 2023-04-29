import tensorflow as tf

model = tf.keras.models.load_model('model.h5')


def predict(img):
    img = tf.keras.preprocessing.image.load_img(
        "test.png", target_size=(32, 32))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    result = model.predict(img, verbose=0)
    return result.argmax()
