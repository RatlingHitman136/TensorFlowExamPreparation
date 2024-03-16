import tensorflow as tf


class AccuracyStopCallback(tf.keras.callbacks.Callback):
    accuracyThreshold = None

    def __init__(self, accuracy_threshold):
        self.accuracyThreshold = accuracy_threshold

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > self.accuracyThreshold:
            self.model.stop_training = True
