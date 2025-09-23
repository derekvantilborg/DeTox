"""
Code to clean up SMILES strings

- clean_mols: Cleans up a molecule
- has_unfamiliar_tokens: Check if a SMILES string has unfamiliar tokens
- flatten_stereochemistry: Get rid of stereochemistry in a SMILES string
- desalter: Get rid of salt from SMILES strings
- remove_common_solvents: Get rid of some of the most commonly used solvents in a SMILES string
- unrepeat_smiles: If a SMILES string contains repeats of the same molecule, return a single one of them
- sanitize_mols: Sanitize a molecules with RDkit
- neutralize_mols: Use pre-defined reactions to neutralize charged molecules

Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""

from typing import Union
import warnings
from tqdm import tqdm
from src.chemoinformatics.utils import flatten_stereochemistry, desalter, sanitize_mol, neutralize_mol, canonicalize_smiles, smiles_to_mols
from src.chemoinformatics.descriptors import ecfpable
from typing import Optional

# simple default pipeline (uses functions defined later in this file)
DEFAULT_CLEANING_STEPS = [
    flatten_stereochemistry,
    desalter,
    sanitize_mol,
    neutralize_mol,
    canonicalize_smiles,
    ecfpable
]

class SmilesCleaner:
    """Barebones pipeline: _base_clean + steps."""
    def __init__(self, steps: Optional[list] = None, *, use_defaults: bool = True):
        if steps is None:
            steps = list(DEFAULT_CLEANING_STEPS) if use_defaults else []
        elif use_defaults:
            steps = list(DEFAULT_CLEANING_STEPS) + list(steps)
        else:
            steps = list(steps)
        self._steps = [self._base_clean] + steps
        self.reasons = {}  # original_input -> step_name

    def _base_clean(self, smiles) -> Optional[str]:
        if smiles is None:
            return None
        if isinstance(smiles, float):
            return None
        if not isinstance(smiles, str):
            return None
        s = smiles.strip()
        if not s:
            return None
        return s

    def clean_single_smiles(self, smiles, verbose: bool = False) -> Optional[str]:
        orig = smiles
        s = smiles
        for step in self._steps:
            try:
                s = step(s)
            except Exception as exc:
                self.reasons[orig] = f"{step.__name__}:exception"
                if verbose:
                    print(f"Exception in {step.__name__} for {orig!r}: {exc}")
                return None
            if s is None:
                self.reasons[orig] = step.__name__
                if verbose:
                    print(f"Failed {orig!r} at {step.__name__}")
                return None
        return s

    def clean_smiles_bulk(self, smiles_list, verbose: bool = False) -> list:
        cleaned = []
        for smi in tqdm(smiles_list):
            out = self.clean_single_smiles(smi, verbose=verbose)
            if out is not None:
                cleaned.append(out)
        return cleaned
    
