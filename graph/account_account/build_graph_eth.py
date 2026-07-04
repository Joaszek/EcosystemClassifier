"""
build_graph_eth.py — address<->address graph from ETHdata (Kaggle real-cats).

Reads raw per-address transaction files from ETHdata/:
  {address}/normal_txs.csv      : ETH transfers (from, to, value, gas, ...)
  {address}/erc20_txs.csv       : ERC20 token transfers
  {address}/erc721_txs.csv      : ERC721 token transfers
  {address}/erc1155_txs.csv     : ERC1155 token transfers

Node-ID space:
  All unique addresses encountered across BE/CE + raw tx files.
  Only BE/CE addresses get labels (y); others are unlabeled context nodes.

Edge features [7]:
  [value_eth, gas_used, is_erc20, is_erc721, is_erc1155, is_normal, is_internal]

Output: graph_eth.pt  (same interface as graph.pt from build_graph.py)
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

DATA_DIR     = "./real-cats"
ETHDATA_DIR  = "./ETHdata"   # path to extracted ETHdata/ from Kaggle
EDGE_ATTR_DIM = 7

NON_NUMERIC_ID_COLS = [
    'max_sent_transaction_id', 'min_sent_transaction_id',
    'max_received_transaction_id', 'min_received_transaction_id',
]
DATETIME_COLS    = ['first_time', 'last_time']
VERIFICATION_COLS = ['etherscan_checked', 'watchback_checked', 'gs_checked']


# ---------------------------------------------------------------------------
# 1. Labeled address table (BE + CE only)
# ---------------------------------------------------------------------------

def load_address_table(data_dir: str) -> pd.DataFrame:
    be = pd.read_csv(f"{data_dir}/BE.tsv", sep="\t")
    ce = pd.read_csv(f"{data_dir}/CE.tsv", sep="\t")
    be['binary_label'] = 0
    ce['binary_label'] = 1
    df = pd.concat([be, ce], ignore_index=True)
    df = df[df['transaction_number'] > 0]
    df = df.drop_duplicates(subset='address').reset_index(drop=True)
    return df


def build_address_features(df: pd.DataFrame):
    df = df.copy()
    for col in DATETIME_COLS:
        dt = pd.to_datetime(df[col], errors='coerce')
        df[col] = (dt.astype('int64') // 10**9).where(dt.notna(), -1)
    drop = ['address', 'label', 'binary_label'] + NON_NUMERIC_ID_COLS + VERIFICATION_COLS
    feat_cols = [c for c in df.columns if c not in drop]
    X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    x = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(df['binary_label'].values, dtype=torch.long)
    address_to_id = {a.lower(): i for i, a in enumerate(df['address'])}
    return x, y, address_to_id, feat_cols


# ---------------------------------------------------------------------------
# 2. Parse ETHdata raw tx files -> edge list
# ---------------------------------------------------------------------------

def safe_read(path: str, cols: list) -> pd.DataFrame:
    """Read CSV, return empty frame if file missing or unreadable."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path, low_memory=False)
        for c in cols:
            if c not in df.columns:
                df[c] = 0
        return df[cols]
    except Exception:
        return pd.DataFrame(columns=cols)


def parse_eth_address(addr_folder: str) -> pd.DataFrame:
    """
    Returns edge rows: [from, to, value_eth, gas_used,
                        is_erc20, is_erc721, is_erc1155, is_normal, is_internal]
    """
    rows = []

    # normal ETH transfers
    df = safe_read(f"{addr_folder}/normal_txs.csv",
                   ['from', 'to', 'value', 'gasUsed', 'isError'])
    df = df[df['isError'].astype(str) == '0']
    for _, r in df.iterrows():
        rows.append((str(r['from']).lower(), str(r['to']).lower(),
                     float(r['value']) / 1e18, float(r['gasUsed']),
                     0, 0, 0, 1, 0))

    # ERC20
    df = safe_read(f"{addr_folder}/erc20_txs.csv",
                   ['from', 'to', 'value', 'gasUsed'])
    for _, r in df.iterrows():
        rows.append((str(r['from']).lower(), str(r['to']).lower(),
                     float(r['value']), float(r['gasUsed']),
                     1, 0, 0, 0, 0))

    # ERC721
    df = safe_read(f"{addr_folder}/erc721_txs.csv",
                   ['from', 'to', 'gasUsed'])
    for _, r in df.iterrows():
        rows.append((str(r['from']).lower(), str(r['to']).lower(),
                     0.0, float(r['gasUsed']),
                     0, 1, 0, 0, 0))

    # ERC1155
    df = safe_read(f"{addr_folder}/erc1155_txs.csv",
                   ['from', 'to', 'gasUsed'])
    for _, r in df.iterrows():
        rows.append((str(r['from']).lower(), str(r['to']).lower(),
                     0.0, float(r['gasUsed']),
                     0, 0, 1, 0, 0))

    return pd.DataFrame(rows, columns=[
        'from', 'to', 'value_eth', 'gas_used',
        'is_erc20', 'is_erc721', 'is_erc1155', 'is_normal', 'is_internal'
    ])


# ---------------------------------------------------------------------------
# 3. Build graph
# ---------------------------------------------------------------------------

def build_graph(data_dir: str = DATA_DIR, ethdata_dir: str = ETHDATA_DIR):
    addr_df = load_address_table(data_dir)
    x_labeled, y, labeled_to_id, feat_cols = build_address_features(addr_df)
    N_labeled = x_labeled.size(0)

    print(f"Labeled addresses: {N_labeled}  (BE+CE)")
    print(f"Scanning ETHdata folders...")

    all_edges = []
    folders = [f for f in os.listdir(ethdata_dir)
               if os.path.isdir(os.path.join(ethdata_dir, f))]
    for i, folder in enumerate(folders):
        if i % 1000 == 0:
            print(f"  {i}/{len(folders)}...")
        edges = parse_eth_address(os.path.join(ethdata_dir, folder))
        all_edges.append(edges)

    edges = pd.concat(all_edges, ignore_index=True)
    edges = edges.dropna(subset=['from', 'to'])
    edges = edges[edges['from'] != edges['to']]  # drop self-loops
    print(f"Raw edges: {len(edges)}")

    # Build global address map (labeled + unlabeled context nodes)
    all_addrs = set(edges['from']) | set(edges['to']) | set(labeled_to_id.keys())
    # labeled addresses get ids 0..N_labeled-1 (preserving order)
    global_to_id = dict(labeled_to_id)
    next_id = N_labeled
    for addr in sorted(all_addrs):
        if addr not in global_to_id:
            global_to_id[addr] = next_id
            next_id += 1
    N_total = next_id
    N_context = N_total - N_labeled
    print(f"Context (unlabeled) nodes: {N_context}")
    print(f"Total nodes: {N_total}")

    # Node features: labeled get real features, context get zeros
    x = torch.zeros((N_total, x_labeled.size(1)), dtype=torch.float32)
    x[:N_labeled] = x_labeled

    # Labels: labeled get real labels, context get -1 (ignored in loss)
    y_full = torch.full((N_total,), -1, dtype=torch.long)
    y_full[:N_labeled] = y

    # Edge tensors
    src = edges['from'].map(global_to_id).dropna().to_numpy(dtype=np.int64)
    dst = edges['to'].map(global_to_id).dropna().to_numpy(dtype=np.int64)
    valid = (edges['from'].isin(global_to_id)) & (edges['to'].isin(global_to_id))
    edges = edges[valid]
    src = edges['from'].map(global_to_id).to_numpy(dtype=np.int64)
    dst = edges['to'].map(global_to_id).to_numpy(dtype=np.int64)

    attr_cols = ['value_eth', 'gas_used',
                 'is_erc20', 'is_erc721', 'is_erc1155', 'is_normal', 'is_internal']
    attr = edges[attr_cols].to_numpy(dtype=np.float32)

    # Directed graph (from -> to) -- we have real direction from raw tx
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_attr  = torch.tensor(attr, dtype=torch.float32)

    data = Data(x=x, y=y_full, edge_index=edge_index, edge_attr=edge_attr)
    data.N_labeled  = N_labeled
    data.N_context  = N_context
    data.num_feat   = len(feat_cols)
    data.EDGE_ATTR_DIM = EDGE_ATTR_DIM

    # Stratified split on labeled nodes only
    idx    = np.arange(N_labeled)
    y_np   = y.numpy()
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y_np, random_state=42)
    train_idx, val_idx  = train_test_split(
        train_idx, test_size=0.1, stratify=y_np[train_idx], random_state=42)
    for split_name, ids in [('train_mask', train_idx),
                             ('val_mask',   val_idx),
                             ('test_mask',  test_idx)]:
        mask = torch.zeros(N_total, dtype=torch.bool)
        mask[ids] = True
        setattr(data, split_name, mask)

    return data, global_to_id, feat_cols


if __name__ == "__main__":
    data, global_to_id, feat_cols = build_graph()
    print(data)
    print(f"N_labeled  = {data.N_labeled}")
    print(f"N_context  = {data.N_context}")
    print(f"feat_dim   = {data.num_feat}")
    print(f"edge_attr  = {data.edge_attr.shape}  "
          f"(value_eth, gas_used, is_erc20, is_erc721, is_erc1155, is_normal, is_internal)")
    print(f"label counts: {torch.bincount(data.y[data.y >= 0]).tolist()}")
    print(f"train/val/test: {data.train_mask.sum().item()}/"
          f"{data.val_mask.sum().item()}/{data.test_mask.sum().item()}")
    torch.save(data, "graph_eth.pt")
    print("Saved -> graph_eth.pt")