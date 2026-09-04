import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

MAX_NODES = 5000
MAX_NEIGHBORS = MAX_NODES - 1


def extract_subgraph(data: Data, center_idx: int,
                     num_hops: int = 1,
                     seed: int = None) -> Data:

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=center_idx,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.N_addr + data.N_tok,
    )

    was_capped = False

    if subset.numel() > MAX_NODES:
        was_capped = True
        rng = np.random.default_rng(seed if seed is not None else center_idx)

        is_center = (subset == center_idx)
        token_mask = ~(subset < data.N_addr)
        token_positions = token_mask.nonzero(as_tuple=True)[0]

        n_keep = min(MAX_NEIGHBORS, token_positions.numel())
        keep_pos = rng.choice(token_positions.numpy(), size=n_keep, replace=False)
        keep_pos = torch.tensor(keep_pos, dtype=torch.long)

        center_local = is_center.nonzero(as_tuple=True)[0]
        kept_local = torch.cat([center_local, keep_pos])
        kept_local_set = set(kept_local.tolist())

        src, dst = edge_index
        edge_keep = torch.tensor(
            [i for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist()))
             if s in kept_local_set and d in kept_local_set],
            dtype=torch.long
        )

        old_to_new = {old: new for new, old in enumerate(kept_local.tolist())}
        subset = subset[kept_local]
        edge_index = torch.stack([
            torch.tensor([old_to_new[s.item()] for s in edge_index[0][edge_keep]]),
            torch.tensor([old_to_new[d.item()] for d in edge_index[1][edge_keep]]),
        ])

        original_edge_mask_indices = edge_mask.nonzero(as_tuple=True)[0]
        edge_mask = original_edge_mask_indices[edge_keep]

        mapping = torch.tensor(
            [old_to_new[center_local[0].item()]], dtype=torch.long
        )

    is_addr = subset < data.N_addr
    n = subset.numel()

    x = torch.zeros((n, data.x.size(1)), dtype=data.x.dtype)
    x[is_addr] = data.x[subset[is_addr]]

    node_type = (~is_addr).long()
    token_global_id = torch.full((n,), -1, dtype=torch.long)
    token_global_id[~is_addr] = subset[~is_addr] - data.N_addr

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=data.edge_attr[edge_mask],
        node_type=node_type,
        token_global_id=token_global_id,
        y=data.y[center_idx].view(1),
        center_idx=mapping.view(1),
        was_capped=torch.tensor([was_capped]),
    )


def build_subgraphs(data: Data, indices, num_hops: int = 2):
    result = []
    n = len(indices)
    for i, idx in enumerate(indices):
        if i % 2000 == 0:
            print(f"    {i}/{n}...")
        result.append(extract_subgraph(data, int(idx), num_hops))
    return result


if __name__ == "__main__":
    data = torch.load("graph.pt", weights_only=False)

    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    val_idx   = data.val_mask.nonzero(as_tuple=True)[0]
    test_idx  = data.test_mask.nonzero(as_tuple=True)[0]

    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"\n{split}:")
        subgraphs = build_subgraphs(data, idx, num_hops=2)
        sizes = torch.tensor([s.num_nodes for s in subgraphs], dtype=torch.float32)
        n_capped = sum(s.was_capped.item() for s in subgraphs)
        n_isolated = int((sizes == 1).sum())
        print(f"  {len(subgraphs)} subgraphs | "
              f"avg_nodes={sizes.mean():.2f} | max={int(sizes.max())} | "
              f"capped={n_capped} | isolated={n_isolated} ({100*n_isolated/len(subgraphs):.1f}%)")
        torch.save(subgraphs, f"subgraphs_2_{split}.pt")
        print(f"  saved -> subgraphs_2_{split}.pt")