
from typing import Union
import numpy as np
from tqdm.auto import tqdm
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import rdFingerprintGenerator
from src.chemoinformatics.cats import cats
from src.chemoinformatics.utils import smiles_to_mols


def rdkit_to_array(fp: list) -> np.ndarray:
    """ Convert a list of RDkit fingerprint objects into a numpy array """
    output = []
    for f in fp:
        arr = np.zeros((1,))
        ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)


def ecfpable(smiles: str):
    """ Check if a SMILES string can be converted to an ECFP fingerprint (i.e., is a valid molecule) """
    if smiles is None:
        return None
    try:
        mol = smiles_to_mols(smiles)
        if mol is None:
            return None
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = mfpgen.GetFingerprint(mol)
        if fp is None:
            return None
        return smiles
    except:
        return None


def mols_to_ecfp(mols: Union[list[Mol], Mol], radius: int = 2, nbits: int = 2048, progressbar: bool = False,
                 to_array: bool = False) -> Union[list, np.ndarray]:
    """ Get ECFPs from a list of RDKit molecule objects

    :param mols: list of RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param radius: Radius of the ECFP (default = 2)
    :param nbits: Number of bits (default = 2048)
    :param progressbar: toggles progressbar (default = False)
    :param to_array: Toggles conversion of RDKit fingerprint objects to a Numpy Array (default = False)
    :return: list of RDKit ECFP fingerprint objects, or a Numpy Array of ECFPs if to_array=True
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    was_list = True if type(mols) is list else False
    mols = mols if was_list else [mols]
    fp = [mfpgen.GetFingerprint(m) for m in tqdm(mols, disable=not progressbar)]
    if not to_array:
        return fp if was_list else fp[0]
    return rdkit_to_array(fp)


def mols_to_cats(mols: Union[list[Mol], Mol], progressbar: bool = False) -> Union[list, np.ndarray]:
    """ Get CATs pharmacophore descriptors from a list of RDKit molecule objects (implementation from Alex Müller)

    Descriptions of the individual features can be obtained with ``cheminformatics.cats.get_cats_sigfactory``.

    :param mols: list of RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :param progressbar: toggles progressbar (default = False)
    :return: a Numpy Array of CATs
    """

    was_list = True if type(mols) is list else False
    mols = mols if was_list else [mols]
    cats_list = [cats(m) for m in tqdm(mols, disable=not progressbar)]

    return np.array(cats_list)


