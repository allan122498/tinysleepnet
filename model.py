import os
import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger("default_log")

class TinySleepNet(tf.keras.Model):
    def __init__(self, config, output_dir="./output", use_rnn=False, use_best=False):
        super(TinySleepNet, self).__init__()
        self.config = config
        self.output_dir = output_dir
        self.use_rnn = use_rnn

        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")

        self.global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int64)
        self.global_epoch = tf.Variable(0, trainable=False, name="global_epoch", dtype=tf.int64)

        # Define inputs
        self.inputs_layer = tf.keras.layers.Input(shape=(self.config["input_size"], 1), name="signals")

        # Build CNN model
        net = self.build_cnn(self.inputs_layer)

        # Append RNN layers if required
        if self.use_rnn:
            net = self.append_rnn(net)

        # Final dense layers for classification
        self.logits = tf.keras.layers.Dense(self.config["n_classes"], activation=None, name="softmax_linear")(net)
        self.preds = tf.keras.layers.Activation("softmax")(self.logits)

        # Define loss and optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"])

        # Compile model
        super().__init__(inputs=self.inputs_layer, outputs=self.preds)
        self.compile(optimizer=self.optimizer, loss=self.loss_object, metrics=["accuracy"])

        # Checkpointing
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=1)

        # Restore model if applicable
        self.restore_model(use_best)

    def restore_model(self, use_best):
        restore_path = self.best_ckpt_path if use_best else self.checkpoint_path
        latest_checkpoint = tf.train.latest_checkpoint(restore_path)
        if latest_checkpoint:
            self.ckpt.restore(latest_checkpoint)
            logger.info(f"Model restored from {latest_checkpoint}")
        else:
            logger.info("Model started from random weights")

    def build_cnn(self, inputs):
        """Builds the CNN portion of TinySleepNet"""
        x = tf.keras.layers.Conv1D(128, kernel_size=int(self.config["sampling_rate"] / 2),
                                   strides=int(self.config["sampling_rate"] / 16), activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=8, strides=8)(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Conv1D(128, kernel_size=8, strides=1, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(128, kernel_size=8, strides=1, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(128, kernel_size=8, strides=1, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=4, strides=4)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        return x  # Return CNN output tensor

    def append_rnn(self, inputs):
        """Appends RNN layers to the CNN output"""
        for _ in range(self.config["n_rnn_layers"]):
            inputs = tf.keras.layers.LSTM(self.config["n_rnn_units"], return_sequences=False)(inputs)
        return inputs

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            loss = self.loss_object(y, preds)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        acc = tf.keras.metrics.sparse_categorical_accuracy(y, preds)
        return {"loss": loss, "accuracy": tf.reduce_mean(acc)}

    def test_step(self, data):
        x, y = data
        preds = self(x, training=False)
        loss = self.loss_object(y, preds)

        acc = tf.keras.metrics.sparse_categorical_accuracy(y, preds)
        return {"loss": loss, "accuracy": tf.reduce_mean(acc)}

    def save_checkpoint(self):
        self.manager.save()
        logger.info(f"Checkpoint saved to {self.checkpoint_path}")


if __name__ == "__main__":
    config = {
        "input_size": 3000,
        "sampling_rate": 100,
        "n_classes": 5,
        "learning_rate": 0.001,
        "n_rnn_layers": 2,
        "n_rnn_units": 128
    }
    model = TinySleepNet(config=config, output_dir="./output/test", use_rnn=True)