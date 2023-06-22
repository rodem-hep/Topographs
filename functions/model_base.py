"""
Implementation of a model base class which implements a custom training loop. Basically
quite a bit of stuff that tf.keras.Model does but a bit more flexible (and faster).
"""
import abc
from time import time
from typing import Tuple, Union

import numpy as np
import tensorflow as tf


class ModelBaseClass(tf.keras.Model):
    """A model base class which implements a custom training cycle."""

    def __init__(self, config: dict, **kwargs):
        """
        Args
        ----
            config:
                Configuration of the model.

        """
        super().__init__(**kwargs)
        self.config = config
        self.log_dir = config["log_dir"]
        self.loss_tracker = tf.keras.metrics.Mean("loss")
        self.my_metrics = None
        self.lr_schedule = None
        self.optimizer = None

    @abc.abstractmethod
    def calculate_loss(self, y_true, y_pred, mask):
        """
        Abstract method to calculate the loss.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def calculate_metrics(self, y_true, y_pred, mask):
        """
        Abstract method to calculate the value of all metrics.
        """
        raise NotImplementedError()

    @tf.function
    def train_step(
        self,
        data: Union[
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]
        ],
    ) -> dict:
        """
        Taking one training step of the model. One batch is passed through the network,
        the weights are updated based on the gradients of the loss, and all metrics are
        updated based on the results of the batch

        Args
        ----
            data:
                Tuple of tf.Tensors with either 2 or 3 entries. The first entry has to
                be the input features of the batch, the second the truth information
                of the batch, and optionally the last weights of some form.

        Returns
        -------
            Dictionary containing the scores of all metrics.

        """
        x_train, y_train = data

        with tf.GradientTape() as tape:
            y_pred = self(x_train, training=True)
            preds, mask, parton_mask = y_pred[:-1], y_pred[-2], y_pred[-1]
            # Compute the loss value
            loss = self.calculate_loss(y_train, preds, mask, parton_mask)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update the loss tracker
        self.loss_tracker.update_state(loss)
        # Update metrics (includes the metric that tracks the loss)
        self.calculate_metrics(y_train, preds, mask, parton_mask)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(
        self,
        data: Union[
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]
        ],
    ) -> dict:
        """
        Taking one evaluation step of the model. One batch is passed through the
        network, and all metrics are updated based on the results of the batch. No
        weights are updated.

        Args
        ----
            data:
                Tuple of tf.Tensors with either 2 or 3 entries. The first entry has to
                be the input features of the batch, the second the truth information
                of the batch, and optionally the last weights of some form.

        Returns
        -------
            Dictionary containing the scores of all metrics.

        """
        # Unpack the data
        x_test, y_test = data
        # Compute predictions
        y_pred = self(x_test, training=False)
        preds, mask, parton_mask = y_pred[:-1], y_pred[-2], y_pred[-1]
        loss = self.calculate_loss(y_test, preds, mask, parton_mask)

        # Update the loss tracker
        self.loss_tracker.update_state(loss)
        # Update the metrics.
        self.calculate_metrics(y_test, preds, mask, parton_mask)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self) -> list:
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker] + list(self.my_metrics)

    def reset_states(self):
        for metric in self.metrics:
            metric.reset_states()

    def get_lr_schedule(self, n_batches: int):
        """
        Create a learning rate schedule class and set the learning rate of the optimizer
        to the base learning rate

        Args
        ----
            n_batches:
                Number of batches defining one epoch.

        Raises
        ------
            ValueError:
                No valid name for a learning rate schedule is provided.
        """
        if self.config["lr_schedule"]["name"].lower() == "clr":
            self.lr_schedule = CyclicLRCustom(
                train_steps_per_epoch=n_batches, **self.config["lr_schedule"]["config"]
            )
            base_lr = self.config["lr_schedule"]["config"]["base_lr"]
        elif self.config["lr_schedule"]["name"].lower() == "cosine":
            self.config["lr_schedule"]["config"]["first_decay_steps"] *= n_batches
            self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                **self.config["lr_schedule"]["config"]
            )
            base_lr = self.config["lr_schedule"]["config"]["initial_learning_rate"]
        else:
            raise ValueError("No LR schedule provided...")

        tf.keras.backend.set_value(self.optimizer.lr, base_lr)

    def epoch_step(
        self,
        dataset: tf.data.Dataset,
        total_steps: int,
        train: bool = True,
    ) -> Tuple[dict, int]:
        """
        Iterate through one epoch training the model

        Args
        ----
            dataset:
                tf dataset containing the data used by the network
            total_steps:
                Total number of training steps taken so far in the training process.
                This value is needed for the learning rate schedules to adjust the
                learning rate after each step.
            train:
                Whether to train the model or not. Training the model also updates the
                learning rate of the optimizer based on the learning rate schedule.

        Returns
        -------
            results_dict:
                Dictionary containing the scores of all metrics
            total_steps:
                Total number of training steps taken so far in the training process.

        """
        for data in dataset:
            results_dict = self.train_step(data) if train else self.test_step(data)
            if train:
                tf.keras.backend.set_value(
                    self.optimizer.lr, self.lr_schedule(total_steps)
                )
                total_steps += 1

        return results_dict, total_steps

    def epoch_print(
        self,
        summary_writer: tf.summary.SummaryWriter,
        results_dict: dict,
        epoch: int,
        train: bool = True,
    ):
        """
        Print the loss and metrics of the epoch for train/val to the screen and save
        all these values into a tensorboard file. If the model is being trained, the
        learning rate is also saved to the tensorboard file after each epoch.

        Args
        ----
            summary_writer:
                Object to write to the tensorboard file.
            results_dict:
                Dictionary containing the scores of all metrics
            epoch:
                Number of the epoch being currently processed.
            train:
                Whether to train the model or not. During the training the learning rate
                is also saved to the tensorboard file.

        """
        results = ""
        with summary_writer.as_default():
            if train:
                tf.summary.scalar(
                    "lr",
                    tf.keras.backend.get_value(self.optimizer.lr),
                    step=epoch + 1,
                )
            for key, value in results_dict.items():
                results += f"{key}: {value.numpy():.4f}, "
                tf.summary.scalar(key, value, step=epoch + 1)
        print(results)
        self.reset_states()

    def custom_fit(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset):
        """
        Implementing the whole training process. Also writes to tensorboard files,
        applies a learning rate schedule, applies early stopping, and saves the weights
        of the best iteration.

        Args
        ----
            train_dataset:
                tf dataset containing the train data
            val_dataset:
                tf dataset containing the validation data

        """
        train_summary_writer = tf.summary.create_file_writer(
            str(self.log_dir / "train_custom")
        )
        val_summary_writer = tf.summary.create_file_writer(
            str(self.log_dir / "val_custom")
        )

        train_dataset = train_dataset.shuffle(
            buffer_size=self.config["batch_size"] * 3
        ).batch(self.config["batch_size"])
        val_dataset = val_dataset.batch(self.config["batch_size"])
        n_batches = tf.data.experimental.cardinality(train_dataset)

        self.optimizer = tf.keras.optimizers.experimental.AdamW()
        self.get_lr_schedule(n_batches)

        with train_summary_writer.as_default():
            tf.summary.scalar(
                "lr",
                tf.keras.backend.get_value(self.optimizer.lr),
                step=0,
            )

        min_loss = np.inf
        best_epoch = 0

        total_steps = 0
        for epoch in range(self.config["n_epochs"]):
            if epoch - best_epoch > 20:
                print("Loss hasn't improved in ~20 epochs. Stopping training!")
                break

            start_time = time()
            print(f"\nStart epoch {epoch + 1}/{self.config['n_epochs']}")

            train_dict, total_steps = self.epoch_step(train_dataset, total_steps, True)
            self.epoch_print(train_summary_writer, train_dict, epoch, True)
            val_dict, _ = self.epoch_step(val_dataset, total_steps, False)
            self.epoch_print(val_summary_writer, val_dict, epoch, False)

            if val_dict["loss"].numpy() < min_loss:
                self.save_weights(
                    self.log_dir
                    / f"model_{epoch + 1:02d}-{val_dict['loss'].numpy():.2f}.h5",
                    save_format="h5",
                )
                print(
                    f"Val loss improved from {min_loss:.2f} to {val_dict['loss'].numpy():.2f},"
                    + "saving weights to "
                    + f"{self.log_dir}/model_{epoch + 1:02d}-{val_dict['loss'].numpy():.2f}.h5"
                )
                min_loss = val_dict["loss"].numpy()
                best_epoch = epoch

            print(f"Time taken: {time() - start_time:.2f}s")


class CyclicLRCustom(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom cyclical learning rate schedule. The learning rate is changed according to
    a triangular shape, starting at the lowest value, rising linearly to the highest
    one, and decreasing linearly back down to the lowest value.
    """

    def __init__(
        self,
        base_lr: float = 0.0001,
        increase_lr: float = 50,
        n_epochs: int = 20,
        train_steps_per_epoch: int = 10000,
    ):
        super().__init__()
        self.base_lr = base_lr
        self.increase_lr = increase_lr
        self.n_epochs = n_epochs
        self.train_steps_per_epoch = train_steps_per_epoch
        self.step_size = self.n_epochs * self.train_steps_per_epoch

    def __call__(self, step: int) -> float:
        cycle = tf.floor(1 + step / (2 * self.step_size))
        cycle_value = tf.abs(step / self.step_size - 2 * cycle + 1)
        new_lr = self.base_lr + (
            (self.base_lr * self.increase_lr) - self.base_lr
        ) * tf.maximum(0, (1 - cycle_value))

        return new_lr

    def get_config(self) -> dict:
        config = {
            "base_lr": self.base_lr,
            "increase_lr": self.increase_lr,
            "n_epochs": self.n_epochs,
            "train_steps_per_epoch": self.train_steps_per_epoch,
        }
        return config
