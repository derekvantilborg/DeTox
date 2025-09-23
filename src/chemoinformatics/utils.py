
from typing import Union
import warnings
from src.constants import NEUTRALIZATION_PATTERNS
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold


def canonicalize_smiles(smiles: Union[str, list[str]]) -> Union[str, list[str]]:
    """ Canonicalize a list of SMILES strings with the RDKit SMILES canonicalization algorithm """
    if type(smiles) is str:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]


def smiles_to_mols(smiles: Union[list[str], str], sanitize: bool = True, partial_charges: bool = False) -> list:
    """ Convert a list of SMILES strings to RDkit molecules (and sanitize them)

    :param smiles: List of SMILES strings
    :param sanitize: toggles sanitization of the molecule. Defaults to True.
    :param partial_charges: toggles the computation of partial charges (default = False)
    :return: List of RDKit mol objects
    """
    mols = []
    was_list = True
    if type(smiles) is str:
        was_list = False
        smiles = [smiles]

    for smi in smiles:
        molecule = Chem.MolFromSmiles(smi, sanitize=sanitize)

        # If sanitization is unsuccessful, catch the error, and try again without
        # the sanitization step that caused the error
        if sanitize:
            flag = Chem.SanitizeMol(molecule, catchErrors=True)
            if flag != Chem.SanitizeFlags.SANITIZE_NONE:
                Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)

        if partial_charges:
            Chem.rdPartialCharges.ComputeGasteigerCharges(molecule)

        mols.append(molecule)

    return mols if was_list else mols[0]


def mols_to_smiles(mols: list[Mol]) -> list[str]:
    """ Convert a list of RDKit molecules objects into a list of SMILES strings"""
    return [Chem.MolToSmiles(m) for m in mols] if type(mols) is list else Chem.MolToSmiles(mols)


def get_scaffold(mol, scaffold_type: str = 'bemis_murcko'):
    """ Get the molecular scaffold from a molecule. Supports four different scaffold types:
            `bemis_murcko`: RDKit implementation of the bemis-murcko scaffold; a scaffold of rings and linkers, retains
            some sidechains and ring-bonded substituents.
            `bemis_murcko_bajorath`: Rings and linkers only, with no sidechains.
            `generic`: Bemis-Murcko scaffold where all atoms are carbons & bonds are single, i.e., a molecular skeleton.
            `cyclic_skeleton`: A molecular skeleton w/o any sidechains, only preserves ring structures and linkers.

    Examples:
        original molecule: 'CCCN(Cc1ccccn1)C(=O)c1cc(C)cc(OCCCON=C(N)N)c1'
        Bemis-Murcko scaffold: 'O=C(NCc1ccccn1)c1ccccc1'
        Bemis-Murcko-Bajorath scaffold:' c1ccc(CNCc2ccccn2)cc1'
        Generic RDKit: 'CC(CCC1CCCCC1)C1CCCCC1'
        Cyclic skeleton: 'C1CCC(CCCC2CCCCC2)CC1'

    :param mol: RDKit mol object
    :param scaffold_type: 'bemis_murcko' (default), 'bemis_murcko_bajorath', 'generic', 'cyclic_skeleton'
    :return: RDKit mol object
    """
    all_scaffs = ['bemis_murcko', 'bemis_murcko_bajorath', 'generic', 'cyclic_skeleton']
    assert scaffold_type in all_scaffs, f"scaffold_type='{scaffold_type}' is not supported. Pick from: {all_scaffs}"

    # designed to match atoms that are doubly bonded to another atom.
    PATT = Chem.MolFromSmarts("[$([D1]=[*])]")
    # replacement SMARTS (matches any atom)
    REPL = Chem.MolFromSmarts("[*]")

    Chem.RemoveStereochemistry(mol)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    if scaffold_type == 'bemis_murcko':
        return scaffold

    if scaffold_type == 'bemis_murcko_bajorath':
        scaffold = AllChem.DeleteSubstructs(scaffold, PATT)
        return scaffold

    if scaffold_type == 'generic':
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return scaffold

    if scaffold_type == 'cyclic_skeleton':
        scaffold = AllChem.ReplaceSubstructs(scaffold, PATT, REPL, replaceAll=True)[0]
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        scaffold = MurckoScaffold.GetScaffoldForMol(scaffold)
        return scaffold


def flatten_stereochemistry(smiles: str) -> str:
    """ Remove stereochemistry from a SMILES string """
    return smiles.replace('@', '')


def desalter(smiles, salt_smarts: str = "[Cl,Na,Mg,Ca,K,Br,Zn,Ag,Al,Li,I,O,N,H]") -> str:
    """ Get rid of salt from SMILES strings, e.g., CCCCCCCCC(O)CCC(=O)[O-].[Na+] -> CCCCCCCCC(O)CCC(=O)[O-]

    :param smiles: SMILES string
    :param salt_smarts: SMARTS pattern to remove all salts (default = "[Cl,Br,Na,Zn,Mg,Ag,Al,Ca,Li,I,O,N,K,H]")
    :return: cleaned SMILES w/o salts
    """
    if '.' not in smiles:
        return smiles

    remover = SaltRemover(defnData=salt_smarts)

    new_smi = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(smiles)))

    return new_smi


def _initialise_neutralisation_reactions() -> list[(str, str)]:
    """ adapted from the rdkit contribution of Hans de Winter """
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in NEUTRALIZATION_PATTERNS]


def sanitize_mol(smiles: str) -> Union[str, None]:
    """ Sanitize a molecules with RDkit

    :param smiles: SMILES string
    :return: SMILES string if sanitized or None if failed sanitizing
    """
    # basic checks on SMILES validity
    mol = Chem.MolFromSmiles(smiles)

    # flags: Kekulize, check valencies, set aromaticity, conjugation and hybridization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL

    if mol is not None:
        sanitize_error = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
        if sanitize_error:
            warnings.warn(sanitize_error)
            return None
    else:
        return None

    return Chem.MolToSmiles(mol)


def neutralize_mol(smiles: str) -> str:
    """ Use several neutralisation reactions based on patterns defined in NEUTRALIZATION_PATTERNS to neutralize charged
    molecules

    :param smiles: SMILES string
    :return: SMILES of the neutralized molecule
    """
    mol = Chem.MolFromSmiles(smiles)

    # retrieves the transformations
    transfm = _initialise_neutralisation_reactions()  # set of transformations

    # applies the transformations
    for i, (reactant, product) in enumerate(transfm):
        while mol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]

    # converts back the molecule to smiles
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    return smiles

