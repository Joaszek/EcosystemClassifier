"""
build_graph_eth.py — address<->address graph from ETHdata (Kaggle real-cats).

Reads raw per-address transaction files from ETHdata/:
  {address}/normal_transactions.json    : ETH transfers
  {address}/ERC_20_transactions.json    : ERC20 token transfers
  {address}/ERC_721_transactions.json   : ERC721 token transfers
  {address}/ERC_1155_transactions.json  : ERC1155 token transfers
  {address}/internal_transactions.json  : internal calls

Node-ID space:
  [0, N_labeled)   : BE/CE addresses with labels + 48-dim features
  [N_labeled, ...)  : context nodes (counterparty addresses from raw tx)
                      zero features, label=-1 (ignored in loss)

Edge features [7]:
  [value_eth, gas_used, is_erc20, is_erc721, is_erc1155, is_normal, is_internal]

Output: graph_eth.pt
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

DATA_DIR      = "./real-cats"
ETHDATA_DIR   = "./ETHdata"
EDGE_ATTR_DIM = 7

NON_NUMERIC_ID_COLS = [
    'max_sent_transaction_id', 'min_sent_transaction_id',
    'max_received_transaction_id', 'min_received_transaction_id',
]
DATETIME_COLS     = ['first_time', 'last_time']
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
# 2. Parse ETHdata JSON files -> edge rows
# ---------------------------------------------------------------------------

def load_json(path: str) -> list:
    """Load JSON file, return empty list if missing or malformed."""
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def parse_eth_address(addr_folder: str) -> list:
    """
    Returns list of edge tuples:
      (from, to, value_eth, gas_used,
       is_erc20, is_erc721, is_erc1155, is_normal, is_internal)
    """
    rows = []

    # normal ETH transfers
    for r in load_json(f"{addr_folder}/normal_transactions.json"):
        if str(r.get('isError', '0')) != '0':
            continue
        frm = str(r.get('from', '')).lower()
        to  = str(r.get('to',   '')).lower()
        if not frm or not to:
            continue
        value    = float(r.get('value', 0)) / 1e18
        gas_used = float(r.get('gasUsed', 0))
        rows.append((frm, to, value, gas_used, 0, 0, 0, 1, 0))

    # ERC20
    for r in load_json(f"{addr_folder}/ERC_20_transactions.json"):
        frm = str(r.get('from', '')).lower()
        to  = str(r.get('to',   '')).lower()
        if not frm or not to:
            continue
        # ERC20 value is in token units; normalize to float, not ETH
        value    = float(r.get('value', 0))
        gas_used = float(r.get('gasUsed', 0))
        rows.append((frm, to, value, gas_used, 1, 0, 0, 0, 0))

    # ERC721
    for r in load_json(f"{addr_folder}/ERC_721_transactions.json"):
        frm = str(r.get('from', '')).lower()
        to  = str(r.get('to',   '')).lower()
        if not frm or not to:
            continue
        gas_used = float(r.get('gasUsed', 0))
        rows.append((frm, to, 0.0, gas_used, 0, 1, 0, 0, 0))

    # ERC1155
    for r in load_json(f"{addr_folder}/ERC_1155_transactions.json"):
        frm = str(r.get('from', '')).lower()
        to  = str(r.get('to',   '')).lower()
        if not frm or not to:
            continue
        gas_used = float(r.get('gasUsed', 0))
        rows.append((frm, to, 0.0, gas_used, 0, 0, 1, 0, 0))

    # internal
    for r in load_json(f"{addr_folder}/internal_transactions.json"):
        if str(r.get('isError', '0')) != '0':
            continue
        frm = str(r.get('from', '')).lower()
        to  = str(r.get('to',   '')).lower()
        if not frm or not to:
            continue
        value    = float(r.get('value', 0)) / 1e18
        gas_used = float(r.get('gasUsed', 0))
        rows.append((frm, to, value, gas_used, 0, 0, 0, 0, 1))

    return rows


# ---------------------------------------------------------------------------
# 3. Build graph
# ---------------------------------------------------------------------------

def build_graph(data_dir: str = DATA_DIR, ethdata_dir: str = ETHDATA_DIR):
    addr_df = load_address_table(data_dir)
    x_labeled, y, labeled_to_id, feat_cols = build_address_features(addr_df)
    N_labeled = x_labeled.size(0)

    print(f"Labeled addresses: {N_labeled}  (BE+CE)")
    print(f"Scanning ETHdata folders...")

    folders = sorted([
        f for f in os.listdir(ethdata_dir)
        if os.path.isdir(os.path.join(ethdata_dir, f))
    ])

    all_rows = []
    for i, folder in enumerate(folders):
        if i % 2000 == 0:
            print(f"  {i}/{len(folders)}...")
        rows = parse_eth_address(os.path.join(ethdata_dir, folder))
        all_rows.extend(rows)

    print(f"Raw edges: {len(all_rows)}")

    if not all_rows:
        raise RuntimeError(
            "Brak krawędzi -- sprawdź ETHDATA_DIR i strukturę plików JSON."
        )

    edges = pd.DataFrame(all_rows, columns=[
        'from', 'to', 'value_eth', 'gas_used',
        'is_erc20', 'is_erc721', 'is_erc1155', 'is_normal', 'is_internal'
    ])
    edges = edges.dropna(subset=['from', 'to'])
    edges = edges[edges['from'] != edges['to']]  # drop self-loops
    # drop null/empty addresses
    edges = edges[edges['from'].str.startswith('0x')]
    edges = edges[edges['to'].str.startswith('0x')]
    print(f"Edges after filtering: {len(edges)}")

    # Build global address map
    all_addrs = set(edges['from']) | set(edges['to']) | set(labeled_to_id.keys())
    global_to_id = dict(labeled_to_id)
    next_id = N_labeled
    for addr in sorted(all_addrs):
        if addr not in global_to_id:
            global_to_id[addr] = next_id
            next_id += 1
    N_total   = next_id
    N_context = N_total - N_labeled
    print(f"Context (unlabeled) nodes: {N_context}")
    print(f"Total nodes: {N_total}")

    # Node features
    x = torch.zeros((N_total, x_labeled.size(1)), dtype=torch.float32)
    x[:N_labeled] = x_labeled

    # Labels: -1 for context nodes (ignored in loss)
    y_full = torch.full((N_total,), -1, dtype=torch.long)
    y_full[:N_labeled] = y

    # Edge tensors
    valid = edges['from'].isin(global_to_id) & edges['to'].isin(global_to_id)
    edges = edges[valid]
    src = edges['from'].map(global_to_id).to_numpy(dtype=np.int64)
    dst = edges['to'].map(global_to_id).to_numpy(dtype=np.int64)

    attr_cols = ['value_eth', 'gas_used',
                 'is_erc20', 'is_erc721', 'is_erc1155', 'is_normal', 'is_internal']
    attr = edges[attr_cols].to_numpy(dtype=np.float32)

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_attr  = torch.tensor(attr, dtype=torch.float32)

    data = Data(x=x, y=y_full, edge_index=edge_index, edge_attr=edge_attr)
    data.N_labeled    = N_labeled
    data.N_context    = N_context
    data.num_feat     = len(feat_cols)
    data.EDGE_ATTR_DIM = EDGE_ATTR_DIM

    # Stratified split on labeled nodes only
    idx   = np.arange(N_labeled)
    y_np  = y.numpy()
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
    print(f"label counts (0=benign, 1=criminal): "
          f"{torch.bincount(data.y[data.y >= 0]).tolist()}")
    print(f"train/val/test: {data.train_mask[:data.N_labeled].sum().item()}/"
          f"{data.val_mask[:data.N_labeled].sum().item()}/"
          f"{data.test_mask[:data.N_labeled].sum().item()}")
    torch.save(data, "graph_eth.pt")
    print("Saved -> graph_eth.pt")