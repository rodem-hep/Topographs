"""
Full Topograph model
"""
from typing import List

import tensorflow as tf

from .custom_graph_layers import CreateMask, GraphBlock
from .custom_losses import ClassificationLoss, RegressionLoss
from .custom_topograph_layers import (
    InitializeHelperNodesTopFixed,
    InitializeHelperNodesW,
    ParticlePredictionMLPs,
    TopoGraphBlock,
    TopoGraphData,
)
from .model_base import ModelBaseClass
from .tools import Dataset


class TopographModel(ModelBaseClass):
    """
    Full Topograph model
    """

    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        self._build(config)
        self._create_metrics()

        self.regression_loss = RegressionLoss(config["regression_loss"])

        self.classification_loss = ClassificationLoss(
            **config["classification_loss"],
        )
        self.persistent_edges = config["persistent_edges"]

    def _create_metrics(self) -> None:
        """
        Create metrics for all outputs. These include regression metrics for the
        initialisation and one after each topograph block, and one classification metric
        after each topograph block. The actual value is not calculated by these members.
        The value of the respective metric will be used to update the state of these
        metrics to keep all the nice tf metric functionality.
        """
        self.metric_initialisation = tf.keras.metrics.Mean("Initialisation")
        self.metric_regression = [
            tf.keras.metrics.Mean(f"Regression_{i}") for i in range(self.n_scores)
        ]
        self.metric_classification = [
            tf.keras.metrics.Mean(f"Classification_{i}") for i in range(self.n_scores)
        ]

        self.my_metrics = (
            [self.metric_initialisation]
            + self.metric_regression
            + self.metric_classification
        )

    def _build(self, config: dict) -> None:
        """
        Create all layers that are needed for the model based on some configuration
        dictionary.
        """
        self.create_mask = CreateMask()

        ########################################################################
        # Initial graph block, jets exchange information and get updated
        ########################################################################
        n_iterations = config["initial_graph_block"].pop("n_iterations")
        self.jet_graph_block = [
            GraphBlock(**config["initial_graph_block"]) for _ in range(n_iterations)
        ]

        ########################################################################
        # W initialization block, Ws get initialized and a regression network
        # for the initialized values is used
        ########################################################################
        initialization_w = config["initialization_w"]
        self.initialize_w = InitializeHelperNodesW(
            first_w_initialization=initialization_w["first_w_initialization"],
            second_w_initialization=initialization_w["second_w_initialization"],
            first_attention_architecture=initialization_w[
                "first_attention_architecture"
            ],
            second_attention_architecture=initialization_w[
                "second_attention_architecture"
            ],
        )
        self.initial_w_regression = ParticlePredictionMLPs(
            architecture=initialization_w["regression_net"], n_particles=2
        )

        ########################################################################
        # Top initialization block, tops get initialized and a regression network
        # for the initialized values is used
        ########################################################################
        initialization_top = config["initialization_top"]
        self.initialize_top = InitializeHelperNodesTopFixed(
            jets_pooling=initialization_top["jets_pooling"],
            attention_net_architecture=initialization_top["attention_net_architecture"],
        )
        self.initial_top_regression = ParticlePredictionMLPs(
            architecture=initialization_top["regression_net"], n_particles=2
        )

        ########################################################################
        # TopoGraph layers
        ########################################################################
        self.n_scores = config["Topograph"].pop("n_iterations")
        self.topo_blocks = [
            TopoGraphBlock(
                config["Topograph"],
                config["edge_classification"],
                config["regression_net"],
            )
            for _ in range(self.n_scores)
        ]

    def call(self, inputs: List[tf.Tensor]):
        """
        One pass through the model.
        """
        data = TopoGraphData(jets=inputs[0], mask=self.create_mask(inputs[0]))

        for layer in self.jet_graph_block:
            data.jets = layer(data.jets, data.mask)

        data.nodes_w = self.initialize_w(data.jets, data.mask)
        initialised_nodes_w = self.initial_w_regression(data.nodes_w)

        data.nodes_top = self.initialize_top([data.jets, data.nodes_w], data.mask)
        initialised_nodes_top = self.initial_top_regression(data.nodes_top)

        data.previous_edges = [None] * 9
        (
            regression_loss_w,
            regression_loss_top,
            classification_loss_w,
            classification_loss_top,
        ) = ([], [], [], [])
        for block in self.topo_blocks:
            (
                data,
                regression_w,
                regression_top,
                classification_w,
                classification_top,
            ) = block(data)
            regression_loss_w.append(regression_w)
            regression_loss_top.append(regression_top)
            classification_loss_w.append(classification_w)
            classification_loss_top.append(classification_top)
            if not self.persistent_edges:
                data.previous_edges = [None] * 9

        return (
            initialised_nodes_w,
            initialised_nodes_top,
            regression_loss_w,
            regression_loss_top,
            classification_loss_w,
            classification_loss_top,
            data.mask,
            inputs[1],
        )

    def custom_fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        flavour_tagging: bool = True,
    ):
        """
        Build tf.data.Datasets for training and validation from the Dataset holding all
        the data and fit them.
        """
        train_x = tf.data.Dataset.from_tensor_slices(
            (train_dataset.jets.get_inputs(flavour_tagging), train_dataset.parton_mask)
        )
        train_y = tf.data.Dataset.from_tensor_slices(
            (
                train_dataset.true_edges_w,
                train_dataset.w_partons.momentum,
                train_dataset.true_edges_top,
                train_dataset.top_partons.momentum,
            )
        )
        val_x = tf.data.Dataset.from_tensor_slices(
            (val_dataset.jets.get_inputs(flavour_tagging), val_dataset.parton_mask)
        )
        val_y = tf.data.Dataset.from_tensor_slices(
            (
                val_dataset.true_edges_w,
                val_dataset.w_partons.momentum,
                val_dataset.true_edges_top,
                val_dataset.top_partons.momentum,
            )
        )

        train_dataset = tf.data.Dataset.zip((train_x, train_y))
        val_dataset = tf.data.Dataset.zip((val_x, val_y))

        self.custom_fit_datasets(train_dataset, val_dataset)

    def custom_fit_datasets(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset):
        """
        Fit the model using datasets by utilising the fit function of the base class.
        """
        super().custom_fit(train_ds, val_ds)

    def calculate_regression_loss(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, parton_mask: tf.Tensor
    ) -> List[tf.Tensor]:
        """
        Using the true and predicted values of the parton properties calculate all
        regression losses.
        """
        initialization_loss = self.regression_loss.calculate(
            y_true[1], y_pred[0], y_true[3], y_pred[1], parton_mask
        )[0]

        regression_losses = []
        for (preds_w, preds_top) in zip(y_pred[2], y_pred[3]):
            regression_losses.append(
                self.regression_loss.calculate(
                    y_true[1], preds_w, y_true[3], preds_top, parton_mask
                )[0]
            )

        return [initialization_loss] + regression_losses

    def calculate_classification_loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        mask: tf.Tensor,
        parton_mask: tf.Tensor,
    ) -> List[tf.Tensor]:
        """
        Using the true and predicted values for all edges and the mask for the jets,
        calculate all classification losses.
        """
        classification_losses = []

        for (preds_w, preds_top) in zip(y_pred[4], y_pred[5]):
            classification_losses.append(
                self.classification_loss.calculate(
                    y_true[0], preds_w, y_true[2], preds_top, mask, parton_mask
                )
            )
        classification_losses[-1] = classification_losses[-1]

        return classification_losses

    def calculate_loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        mask: tf.Tensor,
        parton_mask: tf.Tensor,
    ) -> tf.Tensor:
        """
        Calculate both, regression and classification losses.
        """
        mask = mask[..., None]
        parton_mask = tf.expand_dims(parton_mask, -1)

        regression_losses = self.calculate_regression_loss(y_true, y_pred, parton_mask)
        classification_losses = self.calculate_classification_loss(
            y_true, y_pred, mask, parton_mask
        )

        loss = tf.reduce_sum(regression_losses) + tf.reduce_sum(classification_losses)

        return loss

    def calculate_regression_metrics(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, parton_mask: tf.Tensor
    ) -> None:
        """
        Update the state of all regression metrics.
        """
        self.metric_initialisation.update_state(
            self.regression_loss.calculate(
                y_true[1], y_pred[0], y_true[3], y_pred[1], parton_mask
            )[0]
        )

        for (metric, preds_w, preds_top) in zip(
            self.metric_regression, y_pred[2], y_pred[3]
        ):
            metric.update_state(
                self.regression_loss.calculate(
                    y_true[1], preds_w, y_true[3], preds_top, parton_mask
                )[0]
            )

    def calculate_classification_metrics(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        mask: tf.Tensor,
        parton_mask: tf.Tensor,
    ) -> None:
        """
        Update the state of all classification metrics.
        """
        for (metric, preds_w, preds_top) in zip(
            self.metric_classification, y_pred[4], y_pred[5]
        ):
            metric.update_state(
                self.classification_loss.calculate(
                    y_true[0], preds_w, y_true[2], preds_top, mask, parton_mask
                )
            )

    def calculate_metrics(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        mask: tf.Tensor,
        parton_mask: tf.Tensor,
    ) -> None:
        """
        Update the state of all metrics, regression and classification.
        """
        mask = mask[..., None]
        parton_mask = tf.expand_dims(parton_mask, -1)
        self.calculate_regression_metrics(y_true, y_pred, parton_mask)

        self.calculate_classification_metrics(y_true, y_pred, mask, parton_mask)
