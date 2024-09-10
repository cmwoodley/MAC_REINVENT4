"""Simple parser routines for TOML and JSON

FIXME: about everything
"""

__all__ = ["read_smiles_csv_file", "read_toml", "read_json", "write_json"]
import sys
import csv
import json
from typing import List, Tuple, Union, Optional, Callable

import tomli

from rdkit import Chem

smiles_func = Callable[[str], str]


def has_multiple_attachment_points_to_same_atom(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        raise RuntimeError(f"Error: Input {smiles} is not a valid molecule")

    seen = set()

    for atom in mol.GetAtoms():
        if atom.HasProp("dummyLabel"):
            neighbours = atom.GetNeighbors()

            if len(neighbours) > 1:
                raise RuntimeError("Error: dummy atom is not terminal")

            idx = neighbours[0].GetIdx()

            if idx in seen:
                return True

            seen.add(idx)

    return False


def read_smiles_csv_file(
    filename: str,
    columns: Union[int, slice],
    delimiter: str = "\t",
    header: bool = False,
    actions: List[smiles_func] = None,
    remove_duplicates: bool = False,
) -> Union[List[str], List[Tuple]]:
    """Read a SMILES column from a CSV file

    FIXME: needs to be made more robust

    :param filename: name of the CSV file
    :param columns: what number of the column to extract
    :param delimiter: column delimiter, must be a single character
    :param header: whether a header is present
    :param actions: a list of callables that act on each SMILES (only Reinvent
                    and Mol2Mol)
    :param remove_duplicates: whether to remove duplicates
    :returns: a list of SMILES or a list of a tuple of SMILES
    """

    macrocycle = False
    smilies = []
    frontier = set()

    with open(filename, "r") as csvfile:
        if header:
            csvfile.readline()

        reader = csv.reader(csvfile, delimiter=delimiter)

        for row in reader:
            stripped_row = "".join(row).strip()

            if not stripped_row or stripped_row.startswith("#"):
                continue

            if isinstance(columns, int):
                smiles = row[columns].strip()
                orig_smiles = smiles

                if actions:
                    for action in actions:
                        if callable(action) and smiles:
                            smiles = action(orig_smiles)

                if not smiles:
                    continue

                # lib/linkinvent
                if "." in smiles:  # assume this is the standard SMILES fragment separator
                    smiles = smiles.replace(".", "|")

                if smiles.endswith("MAC"):
                    smiles, reaction = get_warheads_for_macrocycle(smiles[:-3])
                    macrocycle = True

            else:
                smiles = tuple(smiles.strip() for smiles in row[columns])
                tmp_smiles = smiles

                # FIXME: hard input check for libinvent / linkinvent
                #        for unsupported scaffolds containing multiple
                #        attachment points to the same atoms.
                # libinvent
                new_smiles = smiles[1]

                if "." in new_smiles:  # assume this is the standard SMILES fragment separator
                    new_smiles = new_smiles.replace(".", "|")

                if "|" in new_smiles:
                    if has_multiple_attachment_points_to_same_atom(smiles[0]):
                        raise ValueError(
                            f"Not supported: Smiles {new_smiles} contains multiple attachment points for the same atom"
                        )

                    tmp_smiles = (smiles[0], new_smiles)

                # linkinvent
                new_smiles = smiles[0]

                if "." in new_smiles:  # assume this is the standard SMILES fragment separator
                    new_smiles = new_smiles.replace(".", "|")

                if "|" in new_smiles:
                    if has_multiple_attachment_points_to_same_atom(smiles[1]):
                        raise ValueError(
                            f"Not supported: Smiles {new_smiles} contains multiple attachment points for the same atom"
                        )

                    tmp_smiles = (new_smiles, smiles[1])

                smiles = tmp_smiles

            # SMILES transformation may fail
            # FIXME: needs sensible way to report this back to the user
            if smiles:
                if (not remove_duplicates) or (not smiles in frontier):
                    smilies.append(smiles)
                    frontier.add(smiles)

    if macrocycle:
        return smilies, reaction
    else:
        return smilies


def read_toml(filename: Optional[str]) -> dict:
    """Read a TOML file.

    :param filename: name of input file to be parsed as TOML, if None read from stdin
    """

    if isinstance(filename, str):
        with open(filename, "rb") as tf:
            config = tomli.load(tf)
    else:
        config_str = "\n".join(sys.stdin.readlines())
        config = tomli.loads(config_str)

    return config


def read_json(filename: Optional[str]) -> dict:
    """Read JSON file.

    :param filename: name of input file to be parsed as JSON, if None read from stdin
    """

    if isinstance(filename, str):
        with open(filename, "rb") as jf:
            config = json.load(jf)
    else:
        config_str = "\n".join(sys.stdin.readlines())
        config = json.loads(config_str)

    return config


def write_json(data: str, filename: str) -> None:
    """Write data into a JSON file

    :param data: data in a format JSON accepts
    :param filename: output filename
    """
    with open(filename, "w") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=4)

# Function to get SMARTS for a radius around each attachment point
def get_radius_smarts(mol, atom_idx, radius):
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    atoms = set([atom_idx])
    for b in env:
        atoms.add(mol.GetBondWithIdx(b).GetBeginAtomIdx())
        atoms.add(mol.GetBondWithIdx(b).GetEndAtomIdx())
    amap = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    smarts = Chem.MolToSmarts(submol)
    return smarts

# Function to check if all SMARTS patterns are chemically unique
def get_unique_smarts(smarts_list):
    all_unique = True
    for i in range(len(smarts_list)):
        for j in range(i, len(smarts_list)):
            if i == j:
                continue
            else:
                patt_1 = Chem.MolFromSmarts(smarts_list[i])
                patt_2 = Chem.MolFromSmarts(smarts_list[j])

                if patt_1.HasSubstructMatch(patt_2):
                    all_unique = False
                else:
                    continue

    return all_unique

# Function to add atom indices to SMARTS
def add_atom_indices(smart, start_idx=1):
    mol = Chem.MolFromSmarts(smart)
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(start_idx + i)
    return Chem.MolToSmarts(mol).replace("&",""), mol

# Function to get attachment points and bound atoms from the molecule
def get_attachment_points(mol):
    dummy_atoms = []
    real_nums = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 16 and atom.GetFormalCharge() == 1:  # Atom is '[s+]'
            dummy_atoms.append(atom.GetIdx())
            if atom.GetBonds()[0].GetBeginAtom().GetAtomicNum() == 0:
                real_nums.append(atom.GetBonds()[0].GetEndAtomIdx())
            else:
                real_nums.append(atom.GetBonds()[0].GetBeginAtomIdx())

    if len(dummy_atoms) > 2:
        return [],[]
    
    return dummy_atoms, real_nums

# Function to merge fragments into the product
def merge_fragments(fragment_1, fragment_2):
    smarts_product = Chem.MolFromSmiles(f"{fragment_1}.{fragment_2}")
    attachment_points, bound_atoms = get_attachment_points(smarts_product)
    emol = Chem.EditableMol(smarts_product)
    emol.AddBond(bound_atoms[0], bound_atoms[1])
    emol.RemoveAtom(attachment_points[1])
    emol.RemoveAtom(attachment_points[0])

    return Chem.MolToSmarts(emol.GetMol())

def remove_stereochem_smarts(smarts):
    patt = Chem.MolFromSmarts(smarts)
    Chem.RemoveStereochemistry(patt)
    return Chem.MolToSmarts(Chem.MolFromSmiles(Chem.MolToSmiles(patt))).replace("16H2","16")

# Function to introduce sulfur-based dummy atoms and generate a SMIRKS reaction
def get_warheads_for_macrocycle(smiles):
    mol = Chem.MolFromSmiles(smiles)
    attachment_points = [i for i,x in enumerate(mol.GetAtoms()) if x.GetAtomicNum() == 0]

    atom_path = Chem.rdmolops.GetShortestPath(mol, attachment_points[0], attachment_points[1])[1:-1]

    bond_path = [mol.GetBondBetweenAtoms(atom_path[i], atom_path[i+1]) for i in range(len(atom_path)-1)]

    candidate_bonds = [x for x in bond_path if x.GetBondType() == Chem.rdchem.BondType.SINGLE and x.IsInRing() == False]

    for i in range(len(candidate_bonds)):
        emol = Chem.EditableMol(mol)
        bond_to_cleave = candidate_bonds[i]
        emol.RemoveBond(bond_to_cleave.GetBeginAtomIdx(), bond_to_cleave.GetEndAtomIdx())
        
        # Add sulfur-based dummy atoms
        s_atom_1 = Chem.Atom(16)  # Sulfur atom
        s_atom_1.SetFormalCharge(1)  # Positive charge
        s_atom_2 = Chem.Atom(16)  # Sulfur atom
        s_atom_2.SetFormalCharge(1)  # Positive charge
        
        dummy_atom_1 = emol.AddAtom(s_atom_1)  # Add first [s+]
        dummy_atom_2 = emol.AddAtom(s_atom_2)  # Add second [s+]

        disconnect_id1 = emol.AddBond(bond_to_cleave.GetBeginAtomIdx(), dummy_atom_1, Chem.rdchem.BondType.SINGLE) - 1
        disconnect_id2 = emol.AddBond(bond_to_cleave.GetEndAtomIdx(), dummy_atom_2, Chem.rdchem.BondType.SINGLE) - 1
        new_mol = emol.GetMol()

        dis_atom1 = new_mol.GetBondWithIdx(disconnect_id1).GetEndAtomIdx()
        dis_atom2 = new_mol.GetBondWithIdx(disconnect_id2).GetEndAtomIdx()

        env_at1 = get_radius_smarts(new_mol, attachment_points[0], 4)
        env_at2 = get_radius_smarts(new_mol, attachment_points[1], 4)
        env_dis1 = get_radius_smarts(new_mol, dis_atom1, 4)
        env_dis2 = get_radius_smarts(new_mol, dis_atom2, 4)

        smarts_list = [env_at1, env_at2, env_dis1, env_dis2]
        if any("@" in x for x in smarts_list):
            smarts_list = [remove_stereochem_smarts(x) for x in smarts_list]
        all_unique = get_unique_smarts(smarts_list)

        if all_unique:
            break


    env_dis1, mol1 = add_atom_indices(env_dis1, start_idx=1)
    env_dis2, mol2 = add_atom_indices(env_dis2, start_idx=mol1.GetNumAtoms() + 1)

    product_smarts = merge_fragments(env_dis1, env_dis2)

    # Generate the SMIRKS reaction string
    reaction_smirks = f"({env_dis1}.{env_dis2})>>{product_smarts}"

    return Chem.MolToSmiles(new_mol).replace("SH2+","S+").replace(".","|"), reaction_smirks