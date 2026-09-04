import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import get_path

DATA_DIR = get_path("real_cats")
EDGE_ATTR_DIM = 3

STANDARD_ONEHOT = {
    'ERC20':   [1., 0., 0.],
    'ERC721':  [0., 1., 0.],
    'ERC1155': [0., 0., 1.],
}

NON_NUMERIC_ID_COLS = [
    'max_sent_transaction_id', 'min_sent_transaction_id',
    'max_received_transaction_id', 'min_received_transaction_id',
]
DATETIME_COLS = ['first_time', 'last_time']
VERIFICATION_COLS = ['etherscan_checked', 'watchback_checked', 'gs_checked']

ENTRY_RE = re.compile(r"(\d+):(\d+)T(\d+)")


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
    address_to_id = {a: i for i, a in enumerate(df['address'])}
    return x, y, address_to_id, feat_cols


def parse_cell(cell):
    if pd.isna(cell):
        return []
    # count and T parsed but discarded -- semantics unverified
    return [int(i) for i, c, t in ENTRY_RE.findall(cell)]


def parse_ti_file(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    out = []
    for _, r in raw.iterrows():
        for standard in ('ERC20', 'ERC721', 'ERC1155'):
            for token_idx in parse_cell(r[standard]):
                out.append((r['Address'], standard, token_idx))
    return pd.DataFrame(out, columns=['address', 'standard', 'token_idx'])


def build_graph(data_dir: str = DATA_DIR):
    addr_df = load_address_table(data_dir)
    x, y, address_to_id, feat_cols = build_address_features(addr_df)
    N_addr = x.size(0)

    ti = pd.concat([
        parse_ti_file(f"{data_dir}/TI_B.tsv"),
        parse_ti_file(f"{data_dir}/TI_M.tsv"),
    ], ignore_index=True)
    ti = ti[ti['address'].isin(address_to_id)]

    unique_tokens = sorted(ti['token_idx'].unique())
    token_to_id = {idx: N_addr + i for i, idx in enumerate(unique_tokens)}
    N_tok = len(token_to_id)

    src = ti['address'].map(address_to_id).to_numpy(dtype=np.int64)
    dst = ti['token_idx'].map(token_to_id).to_numpy(dtype=np.int64)

    attr = np.array([STANDARD_ONEHOT[s] for s in ti['standard']], dtype=np.float32)

    fwd_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_index = torch.cat([fwd_index, fwd_index.flip(0)], dim=1)
    edge_attr_t = torch.tensor(attr, dtype=torch.float32)
    edge_attr = torch.cat([edge_attr_t, edge_attr_t], dim=0)   # [2E, 3]

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    data.N_addr = N_addr
    data.N_tok = N_tok
    data.num_feat = len(feat_cols)
    data.EDGE_ATTR_DIM = EDGE_ATTR_DIM

    idx = np.arange(N_addr)
    y_np = y.numpy()
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y_np, random_state=42)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, stratify=y_np[train_idx], random_state=42
    )
    for split_name, ids in [('train_mask', train_idx), ('val_mask', val_idx), ('test_mask', test_idx)]:
        mask = torch.zeros(N_addr, dtype=torch.bool)
        mask[ids] = True
        setattr(data, split_name, mask)

    return data, address_to_id, token_to_id, feat_cols


if __name__ == "__main__":
    data, address_to_id, token_to_id, feat_cols = build_graph()
    print(data)
    print(f"N_addr    = {data.N_addr}")
    print(f"N_tok     = {data.N_tok}")
    print(f"feat_dim  = {data.num_feat}")
    print(f"edge_attr = {data.edge_attr.shape}  (dim={data.EDGE_ATTR_DIM}: ERC20, ERC721, ERC1155)")
    print(f"label counts (benign=0, criminal=1): {torch.bincount(data.y).tolist()}")
    print(f"train/val/test: {data.train_mask.sum().item()}/"
          f"{data.val_mask.sum().item()}/{data.test_mask.sum().item()}")
    torch.save(data, "graph.pt")
    print("Saved -> graph.pt")