import tensorflow as tf
from tensorflow import keras
from customStopCallback import AccuracyStopCallback


def fashionMNIST():
    fashion_dataset = tf.keras.datasets.fashion_mnist
    (train_img, train_label), (test_img, test_label) = fashion_dataset.load_data()
    train_img = train_img / 255.0
    test_img = test_img / 255.0

    model = tf.keras.models.Sequential()
    model.add(keras.layers.Input(shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callback = AccuracyStopCallback(0.95)

    model.fit(train_img, train_label, epochs=10, callbacks=[callback])

    model.evaluate(test_img, test_label)


def img_preprocessing():
    #https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip
    #zip must be downloaded and extracted
    dataset = tf.keras.utils.image_dataset_from_directory()


fashionMNIST()
