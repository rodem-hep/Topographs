"""
Custom loss functions used to train Topographs. They use standard loss functions but
symmetrised to account for symmetries between e.g. tops in ttbar decays.
"""
import tensorflow as tf


class ClassificationLoss:
    """
    Loss function for the edge classification of the Topograph models. It uses the
    standard binary crossentropy but in a symmetric way to account for symmetries
    between e.g. tops in ttbar decays. Additionally, true and false edges can be
    weighted based on the amount of true/false edges.
    """

    def __init__(
        self,
        weighted=False,
    ):
        self.weighted = weighted

    def cross_entropy(
        self,
        label: tf.Tensor,
        pred: tf.Tensor,
        true_weight: tf.Tensor,
        false_weight: tf.Tensor,
    ) -> tf.Tensor:
        """
        Implementation of the crossentropy loss including the option to weight true and
        false examples.
        """
        eps = 1e-5
        if self.weighted:
            return -1 * (
                true_weight[:, None, None] * label * tf.math.log(pred + eps)
                + false_weight[:, None, None]
                * (1 - label)
                * tf.math.log(1 - pred + eps)
            )

        return -1 * (
            label * tf.math.log(pred + eps) + (1 - label) * tf.math.log(1 - pred + eps)
        )

    def calc(
        self,
        y_true_w_part: tf.Tensor,
        y_pred_w_part: tf.Tensor,
        y_true_top_part: tf.Tensor,
        y_pred_top_part: tf.Tensor,
        mask_jets: tf.Tensor,
        mask_w_parton: tf.Tensor,
        mask_top_parton: tf.Tensor,
    ) -> tf.Tensor:
        """
        Calculate weights for true and false edges based on the total amount of
        edges/jets to have a balanced class representation overall. This only works if
        all partons are reconstructable. Using these weights (or not), calculates the
        crossentropy for one specified combination of node and parton.
        """
        n_jets = tf.reduce_sum(mask_jets, axis=[1, 2])
        false_weight_w = 2 * (2 / n_jets)
        true_weight_w = 2 - false_weight_w
        false_weight_top = 2 * (1 / n_jets)
        true_weight_top = 2 - false_weight_top

        if mask_jets is not None:
            return tf.reduce_sum(
                (
                    mask_w_parton
                    * mask_jets
                    * self.cross_entropy(
                        y_true_w_part,
                        y_pred_w_part,
                        true_weight_w,
                        false_weight_w,
                    )
                ),
                axis=[1, 2],
            ) / tf.reduce_sum(mask_jets, axis=[1, 2]) + tf.reduce_sum(
                (
                    mask_top_parton
                    * mask_jets
                    * self.cross_entropy(
                        y_true_top_part,
                        y_pred_top_part,
                        true_weight_top,
                        false_weight_top,
                    )
                ),
                axis=[1, 2],
            ) / tf.reduce_sum(
                mask_jets, axis=[1, 2]
            )

        return tf.reduce_mean(
            mask_w_parton
            * self.cross_entropy(
                y_true_w_part,
                y_pred_w_part,
                true_weight_w,
                false_weight_w,
            ),
            axis=[1, 2],
        ) + tf.reduce_mean(
            mask_top_parton
            * self.cross_entropy(
                y_true_top_part,
                y_pred_top_part,
                true_weight_top,
                false_weight_top,
            ),
            axis=[1, 2],
        )

    def calculate(
        self,
        y_true_w: tf.Tensor,
        y_pred_w: tf.Tensor,
        y_true_top: tf.Tensor,
        y_pred_top: tf.Tensor,
        mask: tf.Tensor,
        mask_partons: tf.Tensor,
    ) -> tf.Tensor:
        """
        Calculate the overall loss.
        """
        mask_jets = mask
        mask_partons = tf.expand_dims(mask_partons, 1)

        first_parton_first_node = self.calc(
            y_true_w[..., 0, :],
            y_pred_w[..., 0, :],
            y_true_top[..., 0, :],
            y_pred_top[..., 0, :],
            mask_jets,
            mask_partons[:, :, 0, :],
            mask_partons[:, :, 1, :],
        )
        first_parton_second_node = self.calc(
            y_true_w[..., 0, :],
            y_pred_w[..., 1, :],
            y_true_top[..., 0, :],
            y_pred_top[..., 1, :],
            mask_jets,
            mask_partons[:, :, 0, :],
            mask_partons[:, :, 1, :],
        )
        second_parton_first_node = self.calc(
            y_true_w[..., 1, :],
            y_pred_w[..., 0, :],
            y_true_top[..., 1, :],
            y_pred_top[..., 0, :],
            mask_jets,
            mask_partons[:, :, 2, :],
            mask_partons[:, :, 3, :],
        )
        second_parton_second_node = self.calc(
            y_true_w[..., 1, :],
            y_pred_w[..., 1, :],
            y_true_top[..., 1, :],
            y_pred_top[..., 1, :],
            mask_jets,
            mask_partons[:, :, 2, :],
            mask_partons[:, :, 3, :],
        )

        return tf.reduce_mean(
            tf.math.minimum(
                first_parton_first_node + second_parton_second_node,
                first_parton_second_node + second_parton_first_node,
            )
        )


class RegressionLoss:
    """
    Loss function for the node regression of the Topograph models. It uses a
    standard regression loss but in a symmetric way to account for symmetries
    between e.g. tops in ttbar decays. Options for the loss are: MAE, logcosh, and huber
    """

    def __init__(self, loss):
        if loss.lower() == "mae":
            self.loss = self.mean_absolute_error
        else:
            raise NotImplementedError()

    def mean_absolute_error(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, mask: tf.Tensor
    ) -> tf.Tensor:
        return tf.reduce_mean(mask * tf.abs(y_true - y_pred), axis=-1)

    def calculate(
        self,
        y_true_w: tf.Tensor,
        y_pred_w: tf.Tensor,
        y_true_top: tf.Tensor,
        y_pred_top: tf.Tensor,
        parton_mask: tf.Tensor,
    ) -> tf.Tensor:
        """
        Calculate the overall loss.
        """
        y_pred_w_first_node = tf.convert_to_tensor(y_pred_w[:, 0])
        y_pred_w_second_node = tf.convert_to_tensor(y_pred_w[:, 1])
        y_true_w_first_parton = tf.cast(y_true_w[:, 0], y_pred_w.dtype)
        y_true_w_second_parton = tf.cast(y_true_w[:, 1], y_pred_w.dtype)

        y_pred_top_first_node = tf.convert_to_tensor(y_pred_top[:, 0])
        y_pred_top_second_node = tf.convert_to_tensor(y_pred_top[:, 1])
        y_true_top_first_parton = tf.cast(y_true_top[:, 0], y_pred_top.dtype)
        y_true_top_second_parton = tf.cast(y_true_top[:, 1], y_pred_top.dtype)

        loss_first_parton_is_first_node = (
            self.loss(y_true_w_first_parton, y_pred_w_first_node, parton_mask[:, 0])
            + self.loss(y_true_w_second_parton, y_pred_w_second_node, parton_mask[:, 2])
            + self.loss(
                y_true_top_first_parton, y_pred_top_first_node, parton_mask[:, 1]
            )
            + self.loss(
                y_true_top_second_parton, y_pred_top_second_node, parton_mask[:, 3]
            )
        )
        loss_first_parton_is_second_node = (
            self.loss(y_true_w_first_parton, y_pred_w_second_node, parton_mask[:, 2])
            + self.loss(y_true_w_second_parton, y_pred_w_first_node, parton_mask[:, 0])
            + self.loss(
                y_true_top_first_parton, y_pred_top_second_node, parton_mask[:, 3]
            )
            + self.loss(
                y_true_top_second_parton, y_pred_top_first_node, parton_mask[:, 1]
            )
        )
        return tf.reduce_mean(
            tf.math.minimum(
                loss_first_parton_is_first_node, loss_first_parton_is_second_node
            )
        ), tf.cast(
            loss_first_parton_is_first_node < loss_first_parton_is_second_node,
            tf.float32,
        )
