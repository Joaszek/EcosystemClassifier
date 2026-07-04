"""
extract_subgraphs_eth.py — ego-subgraph extraction for graph_eth.pt.

Identical logic to extract_subgraphs.py but works on the address<->address
directed graph produced by build_graph_eth.py.

Key differences vs extract_subgraphs.py:
  - No token nodes / node_type / token_global_id
  - edge_attr dim=7: [value_eth, gas_used, is_erc20, is_erc721, is_erc1155,
                       is_normal, is_internal]
  - Graph is directed (real from->to from raw tx); k_hop_subgraph still
    works correctly on directed graphs with undirected=False (default)
  - Labeled mask covers only N_labeled nodes; context nodes are included
    in subgraphs as neighbors but never as centers

MAX_NODES = 5000 cap preserved (same rationale).
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

MAX_NODES     = 5000
MAX_NEIGHBORS = MAX_NODES - 1


def extract_subgraph(data: Data, center_idx: int,
                     num_hops: int = 1,
                     seed: int = None) -> Data:

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=center_idx,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.x.size(0),
    )

    was_capped = False

    if subset.numel() > MAX_NODES:
        was_capped = True
        rng = np.random.default_rng(seed if seed is not None else center_idx)

        is_center     = (subset == center_idx)
        center_local  = is_center.nonzero(as_tuple=True)[0]
        other_pos     = (~is_center).nonzero(as_tuple=True)[0]

        n_keep   = min(MAX_NEIGHBORS, other_pos.numel())
        keep_pos = rng.choice(other_pos.numpy(), size=n_keep, replace=False)
        keep_pos = torch.tensor(keep_pos, dtype=torch.long)

        kept_local     = torch.cat([center_local, keep_pos])
        kept_local_set = set(kept_local.tolist())

        src, dst = edge_index
        edge_keep = torch.tensor(
            [i for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist()))
             if s in kept_local_set and d in kept_local_set],
            dtype=torch.long
        )

        old_to_new = {old: new for new, old in enumerate(kept_local.tolist())}
        subset     = subset[kept_local]
        edge_index = torch.stack([
            torch.tensor([old_to_new[s.item()] for s in edge_index[0][edge_keep]]),
            torch.tensor([old_to_new[d.item()] for d in edge_index[1][edge_keep]]),
        ])

        original_edge_mask_indices = edge_mask.nonzero(as_tuple=True)[0]
        edge_mask = original_edge_mask_indices[edge_keep]
        mapping   = torch.tensor([old_to_new[center_local[0].item()]], dtype=torch.long)

    n = subset.numel()
    x = data.x[subset]   # all nodes have real features (or zeros for context)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=data.edge_attr[edge_mask],
        y=data.y[center_idx].view(1),
        center_idx=mapping.view(1),
        was_capped=torch.tensor([was_capped]),
    )


def build_subgraphs(data: Data, indices, num_hops: int = 1):
    result = []
    n = len(indices)
    for i, idx in enumerate(indices):
        if i % 2000 == 0:
            print(f"    {i}/{n}...")
        result.append(extract_subgraph(data, int(idx), num_hops))
    return result


if __name__ == "__main__":
    data = torch.load("graph_eth.pt", weights_only=False)

    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    val_idx   = data.val_mask.nonzero(as_tuple=True)[0]
    test_idx  = data.test_mask.nonzero(as_tuple=True)[0]

    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"\n{split}:")
        subgraphs = build_subgraphs(data, idx, num_hops=1)
        sizes     = torch.tensor([s.num_nodes for s in subgraphs], dtype=torch.float32)
        n_capped  = sum(s.was_capped.item() for s in subgraphs)
        n_isolated = int((sizes == 1).sum())
        print(f"  {len(subgraphs)} subgraphs | "
              f"avg_nodes={sizes.mean():.2f} | max={int(sizes.max())} | "
              f"capped={n_capped} | isolated={n_isolated} ({100*n_isolated/len(subgraphs):.1f}%)")
        torch.save(subgraphs, f"subgraphs_eth_{split}.pt")
        print(f"  saved -> subgraphs_eth_{split}.pt")