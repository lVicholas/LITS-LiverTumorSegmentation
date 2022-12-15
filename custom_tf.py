import os
import numpy as np
import tensorflow as tf

class DiceScore(tf.keras.metrics.Metric):

    def __init__(self, name='dice_score', **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.dice = self.add_weight(name='dice', initializer='zeros')

    def update_state(self, y_true_one_hot, y_prob, sample_weight=None):

        y_true = tf.argmax(y_true_one_hot, axis=-1)
        y_pred = tf.argmax(y_prob, axis=-1)

        correct_tensor = tf.cast(tf.math.equal(y_true, y_pred), tf.int64)
        n_correct = tf.reduce_sum(correct_tensor)

        n = tf.reduce_sum(y_true)
        
        d = tf.cast(2 * n_correct / n, tf.float32)
        self.dice.assign(d)

    def result(self):
        return self.dice

class MaskProgressSave(tf.keras.callbacks.Callback):

    def __init__(self, sample_x, sample_y, mask_progress_dir):
        super().__init__()

        self.x = tf.expand_dims(sample_x, 0)
        self.mask_progress_dir = mask_progress_dir

        img_path = os.path.join(self.mask_progress_dir, 'image')
        np.save(img_path, self.x.numpy().squeeze())

        mask_path = os.path.join(self.mask_progress_dir, 'target_mask')
        np.save(mask_path, sample_y)

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model(self.x).numpy().squeeze().argmax(axis=-1)
        img_path = os.path.join(self.mask_progress_dir, f'epoch_{epoch+1}')
        np.save(img_path, y_pred)

class BatchLoss(tf.keras.callbacks.Callback):

    def __init__(self, log_path):
        super().__init__()

        self.epoch_loss = []
        self.train_loss = []
        self.log_path = log_path

    def on_batch_end(self, batch, logs=None):
        self.epoch_loss.append(logs['loss'])

    def on_epoch_end(self, epoch, logs):
        self.train_loss.append(self.epoch_loss.copy())
        self.epoch_loss.clear()

    def on_train_end(self, logs=None):
        loss = np.concatenate(self.train_loss).astype(str)
        with open(self.log_path, 'w') as f:
            s = '\n'.join(loss)
            f.write(s)