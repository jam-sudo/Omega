"""Convert SMILES strings to molecular graphs for GNN input.

Converts SMILES strings into graph representations suitable for GNN processing.
Uses RDKit for SMILES parsing and PyTorch Geometric for graph data structures
when available, with pure-PyTorch fallbacks for both.
"""

from __future__ import annotations

import torch

# --- Optional dependency flags ---
try:
    from rdkit import Chem
    from rdkit.Chem import rdchem

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    from torch_geometric.data import Batch, Data

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ---- Atom feature constants ----
COMMON_ATOMS = [6, 7, 8, 16, 15, 9, 17, 35, 53]  # C, N, O, S, P, F, Cl, Br, I
NUM_ATOM_TYPES = len(COMMON_ATOMS) + 1  # +1 for "other"
MAX_DEGREE = 5  # 0-5 => 6 values
MAX_FORMAL_CHARGE = 2  # -2 to +2 => 5 values
NUM_HYBRIDIZATION = 5  # sp, sp2, sp3, sp3d, sp3d2
NUM_AROMATICITY = 1  # boolean
MAX_NUM_H = 4  # 0-4 => 5 values

ATOM_FEATURE_DIM = (
    NUM_ATOM_TYPES + (MAX_DEGREE + 1) + 5 + NUM_HYBRIDIZATION + NUM_AROMATICITY + (MAX_NUM_H + 1)
)
# 10 + 6 + 5 + 5 + 1 + 5 = 32

# ---- Bond feature constants ----
NUM_BOND_TYPES = 4  # single, double, triple, aromatic
NUM_CONJUGATED = 1  # boolean
NUM_IN_RING = 1  # boolean
NUM_STEREO = 4  # STEREONONE, STEREOANY, STEREOZ, STEREOE

BOND_FEATURE_DIM = NUM_BOND_TYPES + NUM_CONJUGATED + NUM_IN_RING + NUM_STEREO  # 10


class _FallbackData:
    """Minimal Data-like container when PyTorch Geometric is not installed."""

    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch: torch.Tensor | None = None
        self.num_nodes: int = x.size(0)

    def __repr__(self) -> str:
        return (
            f"FallbackData(x={list(self.x.shape)}, "
            f"edge_index={list(self.edge_index.shape)}, "
            f"edge_attr={list(self.edge_attr.shape)})"
        )


class _FallbackBatch:
    """Minimal Batch-like container when PyTorch Geometric is not installed."""

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch
        self.num_nodes: int = x.size(0)


def _one_hot(idx: int, length: int) -> list[float]:
    """Create a one-hot vector."""
    vec = [0.0] * length
    if 0 <= idx < length:
        vec[idx] = 1.0
    return vec


if HAS_RDKIT:
    _HYBRIDIZATION_MAP = {
        rdchem.HybridizationType.SP: 0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
        rdchem.HybridizationType.SP3D: 3,
        rdchem.HybridizationType.SP3D2: 4,
    }

    _STEREO_MAP = {
        rdchem.BondStereo.STEREONONE: 0,
        rdchem.BondStereo.STEREOANY: 1,
        rdchem.BondStereo.STEREOZ: 2,
        rdchem.BondStereo.STEREOE: 3,
    }

    _BOND_TYPE_MAP = {
        rdchem.BondType.SINGLE: 0,
        rdchem.BondType.DOUBLE: 1,
        rdchem.BondType.TRIPLE: 2,
        rdchem.BondType.AROMATIC: 3,
    }


def _atom_features_rdkit(atom: rdchem.Atom) -> list[float]:
    """Extract atom features using RDKit."""
    atomic_num = atom.GetAtomicNum()
    if atomic_num in COMMON_ATOMS:
        atom_type_idx = COMMON_ATOMS.index(atomic_num)
    else:
        atom_type_idx = len(COMMON_ATOMS)  # "other"

    features: list[float] = []
    features.extend(_one_hot(atom_type_idx, NUM_ATOM_TYPES))
    features.extend(_one_hot(min(atom.GetDegree(), MAX_DEGREE), MAX_DEGREE + 1))
    charge = atom.GetFormalCharge()
    charge_idx = max(0, min(charge + MAX_FORMAL_CHARGE, 2 * MAX_FORMAL_CHARGE))
    features.extend(_one_hot(charge_idx, 2 * MAX_FORMAL_CHARGE + 1))
    hyb_idx = _HYBRIDIZATION_MAP.get(atom.GetHybridization(), 2)  # default sp3
    features.extend(_one_hot(hyb_idx, NUM_HYBRIDIZATION))
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.extend(_one_hot(min(atom.GetTotalNumHs(), MAX_NUM_H), MAX_NUM_H + 1))
    return features


def _bond_features_rdkit(bond: rdchem.Bond) -> list[float]:
    """Extract bond features using RDKit."""
    features: list[float] = []
    bt_idx = _BOND_TYPE_MAP.get(bond.GetBondType(), 0)
    features.extend(_one_hot(bt_idx, NUM_BOND_TYPES))
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    stereo_idx = _STEREO_MAP.get(bond.GetStereo(), 0)
    features.extend(_one_hot(stereo_idx, NUM_STEREO))
    return features


def smiles_to_graph(smiles: str) -> Data | _FallbackData | None:
    """Convert a SMILES string to a molecular graph.

    Parameters
    ----------
    smiles : str
        SMILES string for the molecule.

    Returns
    -------
    Data or _FallbackData or None
        PyG Data object (or fallback) with:
        - x: atom feature matrix (num_atoms, ATOM_FEATURE_DIM)
        - edge_index: COO edge index (2, num_edges*2)  [undirected]
        - edge_attr: bond feature matrix (num_edges*2, BOND_FEATURE_DIM)
        Returns None if SMILES is invalid.

    Raises
    ------
    ImportError
        If RDKit is not installed.
    """
    if not HAS_RDKIT:
        raise ImportError(
            "RDKit is required for SMILES-to-graph conversion. "
            "Install with: pip install rdkit  (or conda install -c conda-forge rdkit)"
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(_atom_features_rdkit(atom))
    x = torch.tensor(atom_feats, dtype=torch.float32)

    # Bond features (undirected: add both directions)
    src_list: list[int] = []
    dst_list: list[int] = []
    bond_feats: list[list[float]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features_rdkit(bond)
        src_list.extend([i, j])
        dst_list.extend([j, i])
        bond_feats.extend([bf, bf])

    if len(src_list) > 0:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(bond_feats, dtype=torch.float32)
    else:
        # Single atom molecule (no bonds)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float32)

    if HAS_PYG:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return _FallbackData(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_graph_batch(
    smiles_list: list[str],
) -> Batch | _FallbackBatch | None:
    """Convert a list of SMILES strings to a batched graph.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.

    Returns
    -------
    Batch or _FallbackBatch or None
        Batched graph data, or None if all SMILES are invalid.
    """
    graphs = []
    for smi in smiles_list:
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)

    if not graphs:
        return None

    if HAS_PYG:
        return Batch.from_data_list(graphs)

    # Manual batching for fallback
    xs: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    edge_attrs: list[torch.Tensor] = []
    batch_indices: list[torch.Tensor] = []
    node_offset = 0

    for i, g in enumerate(graphs):
        xs.append(g.x)
        if g.edge_index.size(1) > 0:
            edge_indices.append(g.edge_index + node_offset)
        else:
            edge_indices.append(g.edge_index)
        edge_attrs.append(g.edge_attr)
        batch_indices.append(torch.full((g.x.size(0),), i, dtype=torch.long))
        node_offset += g.x.size(0)

    return _FallbackBatch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(edge_indices, dim=1)
        if edge_indices
        else torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.cat(edge_attrs, dim=0)
        if edge_attrs
        else torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float32),
        batch=torch.cat(batch_indices, dim=0),
    )
