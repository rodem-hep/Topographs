"""
Classes used throughout the framework.
"""
from __future__ import annotations

import argparse
import copy
import re
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Optional, Union

import joblib
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler


class CustomParser(argparse.ArgumentParser):
    """
    Base Class for an ArgumentParser used througout the framework.
    """

    def __init__(self, **kwargs):
        """
        Initialize the CustomParser.
        """
        super().__init__(**kwargs)

        self.add_argument(
            "config_file",
            action="store",
            type=Path,
            default="configs/config_full.yaml",
            help="Config file to read from.",
        )
        self.add_argument(
            "log_dir",
            action="store",
            type=Path,
            default="test",
            help="Directory where to save outputs.",
        )


class Configuration:
    """Base Class for reading a .yaml configuration file and preparing it for use."""

    def __init__(self, config_file: Path):
        """
        Initializate the Configuration base class.

        Args
        ----
            config_file:
                Path where the configuration file is saved.
        """
        self.config_file = config_file
        self.loader = self.get_loader()
        self.config = self.load_config(self.config_file)

    def get_loader(self) -> yaml.SafeLoader:
        """Create a yaml loader that can read scientific notification like 1e-3."""
        loader = yaml.SafeLoader
        # This is needed to load numbers in scientific notation
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )

        def construct_path(loader, node):
            """Helper function to be able to load pathlib.Path objects from yaml file."""
            return Path("/".join(loader.construct_sequence(node)))

        loader.add_constructor(
            "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath", construct_path
        )

        return loader

    def load_config(self, config_file: Path) -> dict:
        """
        Read the config file.

        Args
        ----
            config_file:
                Path to the config file to read.
        """
        with open(config_file, "r", encoding="utf-8") as yml_file:
            cfg = yaml.load(yml_file, Loader=self.loader)

        return cfg

    def replace_with_command_line_arguments(self, args: argparse.Namespace) -> None:
        """
        Replace items in the configuration with arguments set on the command line.

        Args
        ----
            args:
                Arguments returned from an ArgumentParser. The arguments set on
                the command line have priority over values set in the config file.
        """
        args = vars(args)

        for key in args:
            if args[key] is not None:
                self.config[key] = args[key]

    def dump(self, file_name: str) -> None:
        """
        Save the config to disk.

        Args
        ----
            file_name:
                Path where the config will be saved.
        """
        with open(file_name, "w", encoding="utf-8") as yml_file:
            yaml.dump(self.config, yml_file, default_flow_style=False)


class CustomStandardScaler:
    """
    An extension to sklearn's standard scaler to automatically deal with the data used
    here, e.g. multiple jets where the scaling should be the same for all features.
    Also doesn't include jets that aren't present in the caluclation of mean and
    std_dev. The actual scaling is performed by sklearn's StandardScaler.
    """

    def __init__(self, same_scaler: bool = True, **kwargs):
        self.scaler = StandardScaler(**kwargs)
        self.same_scaler = same_scaler
        self.shape = None

    def transform_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Depending whether the same scaler should be used for several objects of
        the same type (particles) the array is reshaped into a 2d array with
        the needed shape. The original shape is saved to reshape the
        transformed array back to the original shape.
        """
        self.shape = arr.shape
        if self.same_scaler:
            arr = arr.reshape(-1, arr.shape[-1])
        else:
            arr = arr.reshape(-1, arr.shape[-1] * arr.shape[-2])

        return arr

    def fit(self, arr: np.ndarray) -> None:
        """
        Transforms the input array into a 2d array. Fits StandardScaler to the
        transformed array. Objects which first variable is exactly zero are not
        considered in the calculation of the mean and the standard deviation.
        """
        arr = self.transform_array(arr)
        weights = arr[..., 0] != 0
        self.scaler.fit(arr, sample_weight=weights)

    def fit_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Transforms the input array into a 2d array. Fits StandardScaler to the
        transformed array and applies the derived transformation. Objects which
        first variable is exactly zero are not considered in the calculation of
        the mean and the standard deviation.
        """
        arr = self.transform_array(arr)
        weights = arr[..., 0] != 0
        arr = self.scaler.fit_transform(arr, sample_weight=weights)
        arr = (arr * weights[:, np.newaxis]).reshape(self.shape)
        return arr

    def transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Transforms the input array into a 2d array. Applies a previously
        derived transformation of the StandardScaler to the transformed array.
        """
        arr = self.transform_array(arr)
        weights = arr[..., 0] != 0
        arr = self.scaler.transform(arr)
        arr = (arr * weights[:, np.newaxis]).reshape(self.shape)
        return arr

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Transforms the input array into a 2d array. Applies a previously
        derived inverse transformation of the StandardScaler to the transformed
        array.
        """
        arr = self.transform_array(arr)
        weights = arr[..., 0] != 0
        arr = self.scaler.inverse_transform(arr)
        arr = (arr * weights[:, np.newaxis]).reshape(self.shape)
        return arr

    def copy_reduced_scaler(self) -> CustomStandardScaler:
        """
        Copy the scaler but only considering the first three variables. This is useful
        to apply a scaler obtained from jets (which use the four momentum) to partons
        (which only use the three momentum).
        """
        copied_scaler = copy.deepcopy(self)
        copied_scaler.scaler.mean_ = copied_scaler.scaler.mean_[:3]
        copied_scaler.scaler.scale_ = copied_scaler.scaler.scale_[:3]
        copied_scaler.scaler.var_ = copied_scaler.scaler.var_[:3]
        copied_scaler.scaler.n_features_in_ = 3

        return copied_scaler


@dataclass
class ParticleStatus:
    """
    Dataclass describing the status of Particles. Defines whether they are in cartesian
    coordinates, whether only the three momentum is given, whether they are
    preprocessed, and whether energy and pt are log scaled.
    """

    cartesian: bool = True
    is_three_momentum: bool = False
    preprocessed: bool = False
    pt_energy_is_log: bool = False


class Particles:
    """
    Class to hold particles like jets and partons. The momentum can be changed from
    cartesian to spherical coordinates, pt and energy can be log scaled, and the
    features can be preprocessed.
    """

    def __init__(
        self,
        input_array: np.ndarray,
        status: ParticleStatus,
        scaler: Union[str, CustomStandardScaler, Path] = None,
        use_same_scaler: bool = True,
    ):
        if input_array.shape[-1] < 3:
            raise ValueError(
                "Last dimension is less than 3. Can't construct particle out of it."
            )

        self.status = status
        if self.status.is_three_momentum:
            self.momentum = input_array[..., :3]
            self.other_variables = input_array[..., 3:]
        else:
            self.momentum = input_array[..., :4]
            self.other_variables = input_array[..., 4:]

        self.scaler = None
        self.set_scaler(scaler)

        self.use_same_scaler = use_same_scaler

        if not self.status.preprocessed and not self.status.pt_energy_is_log:
            if not self.status.is_three_momentum:
                self.check_positive_pt_energy(3)
            elif not self.status.cartesian:
                self.check_positive_pt_energy(0)

    def mask(self, mask: np.ndarray) -> None:
        self.momentum = self.momentum[mask == 1]
        self.other_variables = self.other_variables[mask == 1]

    def check_positive_pt_energy(self, idx: int) -> None:
        """
        Check whether any values for a given momentum slice are smaller than 0. This
        should never be the case for unpreprocessed not log scaled pt and energy.
        """
        if np.any(self.momentum[..., idx] < 0):
            raise ValueError(
                f"Values of index {idx} smaller than 0 which should never happen"
            )

    def convert_to_cartesian(self) -> None:
        """
        Convert array of momentum from (pT, eta, phi, E) to (px, py, pz, E).
        If energy is not given will leave it out.
        """
        # If the data is already in cartesian coordinates, don't change momentum vector
        if self.status.cartesian:
            return

        p_x = self.momentum[..., 0] * np.cos(
            self.momentum[..., 2]
        )  # px = pT * cos(phi)
        p_y = self.momentum[..., 0] * np.sin(
            self.momentum[..., 2]
        )  # py = pT * sin(phi)
        p_z = self.momentum[..., 0] * np.sinh(
            self.momentum[..., 1]
        )  # pz = pT * sinh(eta)

        # if there are infinites in the momentum, set them back to infinite
        p_x[self.momentum[..., 0] == np.inf] = np.inf
        p_y[self.momentum[..., 0] == np.inf] = np.inf
        p_z[self.momentum[..., 0] == np.inf] = np.inf

        # if at least four variables present (e.g. jets), use energy as energy;
        # if there are only three variables (e.g. Ws) skip the energy
        if self.status.is_three_momentum:
            self.momentum = np.stack([p_x, p_y, p_z], axis=-1)
        else:
            energy = self.momentum[..., 3]
            self.momentum = np.stack([p_x, p_y, p_z, energy], axis=-1)

        self.status.cartesian = True

    def convert_to_spherical(self) -> None:
        """
        Convert array of momentum from (px, py, pz, E) to (pT, eta, phi, E).
        If energy is not given will leave it out.
        """
        # If the data is already in spherical coordinates, don't change momentum vector
        if not self.status.cartesian:
            return

        p_t = np.sqrt(
            self.momentum[..., 0] ** 2 + self.momentum[..., 1] ** 2
        )  # pT = sqrt(px**2 + py**2)
        phi = np.arctan2(
            self.momentum[..., 1], self.momentum[..., 0]
        )  # phi = arctan(py / px)
        eta = np.arcsinh(self.momentum[..., 2] / p_t)  # eta = arcsinh(pz / pT)

        # set everything that was infinite back to infinite
        p_t[self.momentum[..., 0] == np.inf] = np.inf
        eta[self.momentum[..., 0] == np.inf] = np.inf
        phi[self.momentum[..., 0] == np.inf] = np.inf

        # if at least four variables present (e.g. jets), use energy as energy;
        # if there are only three variables (e.g. Ws) skip the energy
        if self.status.is_three_momentum:
            self.momentum = np.stack([p_t, eta, phi], axis=-1)
        else:
            energy = self.momentum[..., 3]
            self.momentum = np.stack([p_t, eta, phi, energy], axis=-1)

        self.status.cartesian = False

    def log_scale_pt_energy(self) -> None:
        """
        Log scale energy and pT.
        """
        # If pT and E are already log scaled, do nothing
        if self.status.pt_energy_is_log:
            return

        # If we have the four momentum log scale the energy
        if not self.status.is_three_momentum:
            self.momentum[..., 3] = np.log(self.momentum[..., 3])

        # If the data is in spherical coordinates also log scale pT
        elif not self.status.cartesian:
            self.momentum[..., 0] = np.log(self.momentum[..., 0])

        # set Nans to 0. Nans can come from e.g. non-existent jets which leads to log(0)
        np.nan_to_num(self.momentum, neginf=0, copy=False)

        self.status.pt_energy_is_log = True

    def inverse_log_scale_pt_energy(self) -> None:
        """
        Unlog scale energy and pT.
        """  # If pT and E are not log scaled, do nothing
        if not self.status.pt_energy_is_log:
            return

        # If we have the four momentum unlog scale the energy
        if not self.status.is_three_momentum:
            self.momentum[..., 3] = np.exp(self.momentum[..., 3])

        # If the data is in spherical coordinates also unlog scale pT
        elif not self.status.cartesian:
            self.momentum[..., 0] = np.exp(self.momentum[..., 0])

        self.status.pt_energy_is_log = False

    def preprocess(self) -> None:
        """
        Preprocess the data with a StandardScaler. If it is already preprocessed
        nothing happens.
        """
        if self.status.preprocessed:
            return

        # If no scaler is provided fit one to the data
        if self.scaler is None:
            self.scaler = CustomStandardScaler(same_scaler=self.use_same_scaler)
            self.momentum = self.scaler.fit_transform(self.momentum)

        # If a scaler is provided use this one. E.g. using a scaler fitted on the train
        # set and apply it to the val set
        else:
            self.momentum = self.scaler.transform(self.momentum)

        self.status.preprocessed = True

    def inverse_preprocess(self) -> None:
        """
        Do the inverse preprocessing to get original values back. If the array
        isn't preprocessed or no scaler is defined for these particles nothing will
        happen.
        """
        if not self.status.preprocessed:
            return
        if self.scaler is None:
            return

        self.momentum = self.scaler.inverse_transform(self.momentum)

        self.status.preprocessed = False

    def dump_scaler(self, filename: str) -> None:
        """
        Save the scaler that was used for the preprocessing. If there is no
        scaler yet, nothing will be saved.

        Args
        ----
            filename:
                Name of the file the scaler is saved under.

        """
        if self.scaler is None:
            return

        joblib.dump(self.scaler, filename)

    def set_scaler(self, scaler: Union[str, CustomStandardScaler, Path]) -> None:
        """
        Set a scaler to use for the preprocessing. This can be either a string
        giving a filename from where to load the scaler or a scaler object
        itself.

        Args
        ----
            scaler:
                Scaler to be used for the preprocessing
        """
        if scaler is not None:
            if isinstance(scaler, (str, PosixPath)):
                self.scaler = joblib.load(scaler)
            else:
                self.scaler = scaler
        else:
            self.scaler = None

    def n_events(self) -> int:
        """
        Return number of total events in the arrays.
        """
        return self.momentum.shape[0]

    def preprocess_fit(self, cartesian: bool, scaler: str) -> None:
        """
        Preprocess the particles by transforming them into cartesian coordinates if
        wanted, log scaling pt and energy, using a Standard Scaler to remove mean and
        divide by std dev, and save the scaler on disk.
        The Standard Scaler is fit to the available data to retrieve mean and std dev.
        """
        if cartesian:
            self.convert_to_cartesian()
        self.log_scale_pt_energy()
        self.preprocess()
        self.dump_scaler(scaler)

    def preprocess_transform(
        self, cartesian: bool, scaler: CustomStandardScaler
    ) -> None:
        """
        Preprocess the particles by loading an already existing Standard Scaler,
        transforming them into cartesian coordinates if wanted, log scaling pt and
        energy, using a Standard Scaler to remove mean and divide by std dev.
        The Standard Scaler is only used to transform the data, not fit to it.
        """
        self.set_scaler(scaler)
        if cartesian:
            self.convert_to_cartesian()
        self.log_scale_pt_energy()
        self.preprocess()

    def get_inputs(self, other_variables: bool = False) -> np.ndarray:
        """
        Get the momentun or if wanted the momentum plus the other variables. Used to
        obtain the inputs to the Topograph
        """
        if other_variables:
            return np.concatenate([self.momentum, self.other_variables], axis=-1)

        return self.momentum

    def input_shape(self, other_variables: bool = False) -> tuple:
        """
        Return the shape of the particles. Can specify if the other variables should be
        included or not.
        """
        momentum_shape = self.momentum.shape
        if not other_variables:
            return momentum_shape

        input_shape = momentum_shape[:-1]
        input_shape += (momentum_shape[-1] + self.other_variables.shape[-1],)
        return input_shape

    def mass(self) -> np.ndarray:
        """
        Calculate the mass of the particles. If only the three momentum is saved can't
        calculate the mass. After the call the particles are in cartesian coordinates.
        """
        if self.status.is_three_momentum:
            raise ValueError("Cannot calculate mass from a three vector")

        if not self.status.cartesian:
            self.convert_to_cartesian()

        mass_value = np.sqrt(
            self.momentum[..., -1] ** 2
            - (
                self.momentum[..., 0] ** 2
                + self.momentum[..., 1] ** 2
                + self.momentum[..., 2] ** 2
            )
        )
        return mass_value


class Dataset:
    """
    Class holding all data needed for the training. This includes: the jets, w/top
    partons (can also be the reconstructed ones from the jets), indices of the jets to
    match them to the partons of the ttbar decay, number of jets, number of b-jets, and
    matchability which encodes which partons of the ttbar decay are present (a jet has
    been matched to it).
    From these variables the true edges for W and tops can be calculated.
    """

    def __init__(
        self,
        jets: Particles = None,
        w_partons: Particles = None,
        top_partons: Particles = None,
        jet_indices: np.ndarray = None,
        n_jets: np.ndarray = None,
        n_b_jets: np.ndarray = None,
        matchability: np.ndarray = None,
    ):
        self.jets = jets
        self.w_partons = w_partons
        self.top_partons = top_partons
        self.jet_indices = jet_indices
        self.n_jets = n_jets
        self.n_b_jets = n_b_jets
        self.matchability = matchability

        self.parton_mask = None
        self.true_edges_w = None
        self.true_edges_top = None

    def calc_parton_mask(self) -> None:
        """
        Get a mask which defines which partons are reconstructable.
        """
        self.parton_mask = np.zeros(shape=(self.matchability.shape[0], 4))
        self.parton_mask[:, 0] = (self.matchability >> 3 & 1) & (
            self.matchability >> 4 & 1
        )
        self.parton_mask[:, 1] = (self.parton_mask[:, 0] == 1) & (
            self.matchability >> 5 & 1
        )
        self.parton_mask[:, 2] = (self.matchability >> 0 & 1) & (
            self.matchability >> 1 & 1
        )
        self.parton_mask[:, 3] = (self.parton_mask[:, 2] == 1) & (
            self.matchability >> 2 & 1
        )

    def mask_fully_impossible_events(self):
        """
        Mask events for training on partial events. Events which have no parton, i.e.
        no W boson or top-quark reconstructable are masked.
        """
        if self.parton_mask is None:
            self.calc_parton_mask()
        mask = np.sum(self.parton_mask, axis=-1) >= 1

        self.jets.mask(mask == 1)
        self.w_partons.mask(mask == 1)
        self.top_partons.mask(mask == 1)
        self.jet_indices = self.jet_indices[mask == 1]
        self.n_jets = self.n_jets[mask == 1]
        self.n_b_jets = self.n_b_jets[mask == 1]
        self.matchability = self.matchability[mask == 1]

        self.parton_mask = self.parton_mask[mask == 1]
        if self.true_edges_w is not None:
            self.true_edges_w = self.true_edges_w[mask == 1]
        if self.true_edges_top is not None:
            self.true_edges_top = self.true_edges_top[mask == 1]

    def calc_truth_edges(self) -> None:
        """
        Get the true edges of both the Ws and the tops.
        """
        self.calc_truth_edges_w()
        self.calc_truth_edges_top()

    def calc_truth_edges_w(self) -> None:
        """
        Get the true edges of the Ws.
        """
        if self.jet_indices is None:
            raise ValueError("No jet indices to calculate truth edges")

        true_wplus_edges = (self.jet_indices == 1) | (self.jet_indices == 2)
        true_wminus_edges = (self.jet_indices == 4) | (self.jet_indices == 5)

        self.true_edges_w = (
            np.concatenate(
                [
                    true_wplus_edges.reshape(-1, self.jet_indices.shape[1], 1),
                    true_wminus_edges.reshape(-1, self.jet_indices.shape[1], 1),
                ],
                axis=-1,
            )
            .reshape((-1, self.jet_indices.shape[1], 2, 1))
            .astype("float32")
        )

    def calc_truth_edges_top(self) -> None:
        """
        Get the true edges of the tops.
        """
        if self.jet_indices is None:
            raise ValueError("No jet indices to calculate truth edges")

        true_top_edges = self.jet_indices == 0
        true_antitop_edges = self.jet_indices == 3

        self.true_edges_top = (
            np.concatenate(
                [
                    true_top_edges.reshape(-1, self.jet_indices.shape[1], 1),
                    true_antitop_edges.reshape(-1, self.jet_indices.shape[1], 1),
                ],
                axis=-1,
            )
            .reshape((-1, self.jet_indices.shape[1], 2, 1))
            .astype("float32")
        )

    def preprocess_fit(
        self,
        log_dir: Path,
        same_scaler_everything: bool = False,
        jet_scaler: Optional[CustomStandardScaler] = None,
        w_scaler: Optional[CustomStandardScaler] = None,
        top_scaler: Optional[CustomStandardScaler] = None,
    ) -> None:
        """
        Preprocess the three Particles held in the dataset (jets, Ws, tops) using their
        respective preprocess function. If scalers are provided, they will be used to
        only transform the features, otherwise a new scaler will be fit to the data.
        """
        if same_scaler_everything:
            if jet_scaler is not None:
                self.jets.preprocess_transform(True, jet_scaler)
            else:
                self.jets.preprocess_fit(True, log_dir / "scaler.Jets")

            particle_scaler = self.jets.scaler.copy_reduced_scaler()
            self.w_partons.preprocess_transform(True, particle_scaler)
            self.top_partons.preprocess_transform(True, particle_scaler)
            if w_scaler is None:
                self.w_partons.dump_scaler(log_dir / "scaler.Ws")
            if top_scaler is None:
                self.top_partons.dump_scaler(log_dir / "scaler.Tops")
        else:
            if jet_scaler is not None:
                self.jets.preprocess_transform(True, jet_scaler)
            else:
                self.jets.preprocess_fit(True, log_dir / "scaler.Jets")

            if w_scaler is not None:
                self.w_partons.preprocess_transform(True, w_scaler)
            else:
                self.w_partons.preprocess_fit(True, log_dir / "scaler.Ws")

            if top_scaler is not None:
                self.top_partons.preprocess_transform(True, top_scaler)
            else:
                self.top_partons.preprocess_fit(True, log_dir / "scaler.Tops")
