#!/bin/bash
set -e

echo "============================================"
echo " Real-CATS + ETHdata setup & graph build"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Zależności
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Instalacja zależności..."

pip install -q kagglehub torch-geometric pandas numpy scikit-learn tqdm

# Wykryj wersję PyTorch i CUDA
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda or '')")

echo "  PyTorch: $TORCH_VERSION  CUDA: $CUDA_VERSION"

# Dobierz tag CUDA dla PyG
case "$CUDA_VERSION" in
    11.8*) CUDA_TAG="cu118" ;;
    12.4*) CUDA_TAG="cu124" ;;
    12.6*) CUDA_TAG="cu126" ;;
    12.8*) CUDA_TAG="cu128" ;;
    12.9*) CUDA_TAG="cu129" ;;
    *)     CUDA_TAG="cu128" ;;
esac

# Opcjonalne przyspieszacze PyG (nie wymagane, ale warto mieć)
pip install -q pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html \
    2>/dev/null || echo "  (opcjonalne biblioteki PyG pominięte)"

python3 -c "import torch, torch_geometric; print(f'  torch={torch.__version__}  pyg={torch_geometric.__version__}')"

# ---------------------------------------------------------------------------
# 2. Pobierz dataset przez kagglehub (bez tokena)
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Pobieranie real-cats przez kagglehub..."

python3 << 'PYEOF2'
import kagglehub, shutil, os

path = kagglehub.dataset_download("lvd312393/real-cats")
print(f"  Cache: {path}")

# TSV -> ./real-cats/
os.makedirs("./real-cats", exist_ok=True)
copied_tsv = 0
for f in os.listdir(path):
    if f.endswith(".tsv"):
        shutil.copy(os.path.join(path, f), "./real-cats/")
        copied_tsv += 1
print(f"  TSV skopiowane: {copied_tsv}")

# ETHdata -> ./ETHdata/
candidates = [
    os.path.join(path, "ETHdata", "ETHdata"),
    os.path.join(path, "ETHdata"),
]
eth_src = None
for c in candidates:
    if os.path.isdir(c) and any(d.startswith("0x") for d in os.listdir(c)):
        eth_src = c
        break

if eth_src:
    print(f"  ETHdata źródło: {eth_src}")
    os.makedirs("./ETHdata", exist_ok=True)
    shutil.copytree(eth_src, "./ETHdata", dirs_exist_ok=True)
    print(f"  ETHdata foldery: {len(os.listdir('./ETHdata'))}")
else:
    print("  UWAGA: ETHdata nie znalezione w cache -- sprawdź strukturę ręcznie")
    print(f"  Zawartość {path}:")
    for item in os.listdir(path):
        print(f"    {item}")
PYEOF2

echo "  real-cats/: $(ls ./real-cats/*.tsv 2>/dev/null | wc -l) plików TSV"
echo "  ETHdata/: $(ls ./ETHdata 2>/dev/null | wc -l) folderów adresów"

# ---------------------------------------------------------------------------
# 3. Buduj graf address<->address
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Budowanie grafu address<->address (graph_eth.pt)..."
python3 graph/build_graph_eth.py

# ---------------------------------------------------------------------------
# 4. Ekstrakcja subgrafów -- hop 1 i hop 2
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Ekstrakcja subgrafów hop=1..."
python3 graph/extract_subgraphs_eth.py --hops 1 --suffix ""

echo ""
echo "[5/5] Ekstrakcja subgrafów hop=2..."
python3 graph/extract_subgraphs_eth.py --hops 2 --suffix "_2hop"

echo ""
echo "============================================"
echo " Gotowe! Pliki wyjściowe:"
ls -lh graph_eth.pt \
       subgraphs_eth_train.pt subgraphs_eth_val.pt subgraphs_eth_test.pt \
       subgraphs_eth_2hop_train.pt subgraphs_eth_2hop_val.pt subgraphs_eth_2hop_test.pt \
       2>/dev/null
echo "============================================"