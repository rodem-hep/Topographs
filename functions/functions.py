"""
Functions used throughout the framework.
"""

import glob
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import numpy.lib.recfunctions

from .tools import Dataset, Particles, ParticleStatus


def load_jets(
    data_file: str, load_points: slice, mask: np.ndarray, max_n_jets: int
) -> Particles:
    """
    Load the jets from an h5 file and return them as an 'Particles' class.

    Args
    ----
        data_file:
            Path to the h5 file to load the jets
        load_points:
            Slice of data points to load.
        mask:
            Numpy array to mask events.
        max_n_jets:
            Only return up to this number of jets. Additional jets will be removed.

    Returns
    -------
        jets:
            Loaded jets
    """
    with h5py.File(data_file, "r") as hf:
        jets = hf["delphes/jets"][load_points]
    jets = numpy.lib.recfunctions.structured_to_unstructured(jets)[mask == 1].reshape(
        -1, 16, 5
    )[:, :max_n_jets, :]
    jets = Particles(
        jets,
        status=ParticleStatus(cartesian=False),
        scaler=None,
        use_same_scaler=True,
    )

    return jets


def load_partons(
    data_file: str,
    load_points: slice,
    mask: np.ndarray,
    pdg_id: int,
    loaded_partons: Optional[np.ndarray] = None,
) -> Particles:
    """
    Load partons from an h5 file and return them as a 'Particles' class.

    Args
    ----
        data_file:
            Path to the h5 file to load the jets
        load_points:
            Slice of data points to load.
        mask:
            Numpy array to mask events.
        pdg_id:
            The PDG id of the partons that should be loaded. Will load the parton with
            + and - pdg_id and concatenate them.
        loaded_partons:
            If the partons dataset is already loaded from the h5 file because e.g. other
            partons are needed, don't reload the dataset but use the one that was
            already loaded.

    Returns
    -------
        partons:
            Loaded partons

    """
    if loaded_partons is None:
        with h5py.File(data_file, "r") as hf:
            loaded_partons = hf["delphes/partons"][load_points]
        loaded_partons = numpy.lib.recfunctions.structured_to_unstructured(
            loaded_partons
        )[:, :][mask == 1]
    # get partons and only select pT, eta, phi
    parton = loaded_partons[:, :, 1:4][loaded_partons[:, :, 0] == pdg_id]
    anti_parton = loaded_partons[:, :, 1:4][loaded_partons[:, :, 0] == -pdg_id]
    partons = np.concatenate([parton, anti_parton], axis=-1).reshape((-1, 2, 3))
    partons = Particles(
        partons,
        status=ParticleStatus(cartesian=False, is_three_momentum=True),
        scaler=None,
        use_same_scaler=True,
    )

    return partons, loaded_partons


def load_and_mask_data(
    data_file: str,
    jet_indices: bool = False,
    jets: bool = False,
    partons_w: bool = False,
    partons_top: bool = False,
    min_n_jets: int = 0,
    min_n_b_jets: int = 0,
    max_n_jets: int = 16,
    matchability: bool = True,
    n_events: int = None,
    events_min_index: int = None,
    jet_scale: bool = False,
    jet_resolution: bool = False,
) -> Dataset:
    """
    Load data into a custom 'Dataset'.

    Args
    ----
        data_file:
            Path to the h5 file to load the data from.
        jet_indices:
            Load the jet indices or not
        jets:
            Load the jets or not
        partons_w:
            Load the W partons or not
        partons_top:
            Load the top partons or not
        min_n_jets:
            Only load events with at least that many jets
        min_n_b_jets:
            Only load events with at least that many b-tagged jets
        max_n_jets:
            Only load events with at most that many jets
        matchability:
            Load only fully matchable events or all events
        n_events:
            Only load that number of events. If additional requirements are placed, such
            as minimum number of jets, matchability etc. the number of events returned
            will be lower as this value specifies the number of events loaded from the
            h5 file.
        events_min_index:
            Index in the h5 dataset to start the loading from.
        jet_scale:
            Scale the jet energy up by 2.5% for all jets
        jet_resolution:
            Smear the jet energy by 5% for all jets

    Returns
    -------
        dataset:
            All requested data in a 'Dataset' class. The number of jets, number of
            b-jets and the matchability are always included in the dataset.


    """
    if n_events is not None and events_min_index is not None:
        load_points = slice(events_min_index, events_min_index + n_events)
    else:
        load_points = slice(events_min_index, n_events)
    partons = None
    # load matchability to only select fully matchable events
    with h5py.File(data_file, "r") as hf:
        match = hf["delphes/matchability"][load_points]
        n_jets = hf["delphes/njets"][load_points]
        n_b_jets = hf["delphes/nbjets"][load_points]

    mask = (n_jets >= min_n_jets) & (n_b_jets >= min_n_b_jets) & (n_jets <= max_n_jets)
    if matchability:
        mask = (mask == 1) & (match == 63)

    # load the jet indices
    if jet_indices:
        with h5py.File(data_file, "r") as hf:
            jets_indices_h5 = hf["delphes/jets_indices"][load_points]
        jets_indices_h5 = jets_indices_h5[mask == 1].reshape(-1, 16)[:, :max_n_jets]
    # load the jets and make them an unstructured np array
    if jets:
        input_jets = load_jets(data_file, load_points, mask, max_n_jets)
        if jet_scale:
            input_jets.momentum[..., -1] = input_jets.momentum[..., -1] * 1.025
        if jet_resolution:
            smear = np.random.normal(
                loc=1, scale=0.05, size=(input_jets.momentum.shape[0], 16)
            )
            smear = np.clip(smear, a_min=0, a_max=None)
            input_jets.momentum[..., -1] = input_jets.momentum[..., -1] * smear

    # load the truth partons and for the moment return only the Ws
    if partons_w:
        w_partons, partons = load_partons(data_file, load_points, mask, 24, partons)

    if partons_top:
        top_partons, partons = load_partons(data_file, load_points, mask, 6, partons)

    match = match[mask == 1]
    n_jets = n_jets[mask == 1]
    n_b_jets = n_b_jets[mask == 1]
    dataset = Dataset(
        jets=input_jets if jets else None,
        w_partons=w_partons if partons_w else None,
        top_partons=top_partons if partons_top else None,
        jet_indices=jets_indices_h5 if jet_indices else None,
        n_jets=n_jets,
        n_b_jets=n_b_jets,
        matchability=match,
    )
    return dataset


def get_best_model_weights(log_dir: Path):
    """
    Load model weights which were last saved to load the best ones.

    Args
    ----
        log_dir:
            Path of the directory where the model files are stored

    Returns
    -------
        latest:
            File name of the h5 file that was saved last which should contain the
            weights of the best model.

    """
    list_of_files = glob.glob(str(log_dir / "*.h5"))
    latest = max(list_of_files, key=os.path.getctime)
    return latest
