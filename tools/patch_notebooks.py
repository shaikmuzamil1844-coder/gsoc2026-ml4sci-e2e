import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_cell_source(nb_path: Path, replacements: dict[int, str]) -> None:
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    for idx, source in replacements.items():
        data["cells"][idx]["source"] = source.splitlines(keepends=True)
    nb_path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


task1_replacements = {
    4: """# Install dependencies when running in a fresh notebook runtime
import importlib.util
import subprocess
import sys

required = ['h5py', 'sklearn', 'matplotlib', 'seaborn']
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]

if missing:
    print(f'Installing missing packages: {missing}')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
else:
    print('Dependencies already available.')
""",
    7: """import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_FILES = [
    'SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5',
    'SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5',
]

def have_task1_data():
    return all(Path(name).exists() for name in DATASET_FILES)

def generate_synthetic_task1(path, label, samples=1500, image_size=32, channels=2):
    import h5py
    import numpy as np

    rng = np.random.default_rng(42 + label)
    images = np.zeros((samples, image_size, image_size, channels), dtype=np.float32)
    targets = np.full(samples, label, dtype=np.int64)
    center = image_size // 2

    for i in range(samples):
        spread = 3.2 if label == 1 else 5.5
        hits = rng.poisson(28 if label == 1 else 20)
        eta = np.clip(rng.normal(center, spread, hits).astype(int), 0, image_size - 1)
        phi = np.clip(rng.normal(center, spread, hits).astype(int), 0, image_size - 1)
        energy = rng.exponential(1.2 if label == 1 else 0.9, hits).astype(np.float32)
        time = rng.normal(0.4 if label == 1 else 0.7, 0.12, hits).astype(np.float32)
        np.add.at(images[i, :, :, 0], (eta, phi), energy)
        np.add.at(images[i, :, :, 1], (eta, phi), np.abs(time))

    with h5py.File(path, 'w') as f:
        f.create_dataset('X', data=images)
        f.create_dataset('y', data=targets)

if have_task1_data():
    print('Task 1 dataset already present.')
else:
    kaggle_user = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    kaggle_ok = bool(kaggle_user and kaggle_key)
    kaggle_cli = shutil.which('kaggle')

    if kaggle_ok:
        if kaggle_cli is None:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'])
            kaggle_cli = shutil.which('kaggle')
        try:
            subprocess.check_call([
                kaggle_cli, 'datasets', 'download', '-d',
                'vishakkbhat/electron-vs-photons-ml4sci', '--force'
            ])
            import zipfile
            with zipfile.ZipFile('electron-vs-photons-ml4sci.zip') as zf:
                zf.extractall('.')
        except Exception as exc:
            print(f'Kaggle download failed: {exc}')

    if not have_task1_data():
        print('Dataset unavailable. Generating a small synthetic fallback dataset for notebook execution...')
        generate_synthetic_task1(DATASET_FILES[0], label=0)
        generate_synthetic_task1(DATASET_FILES[1], label=1)

for name in DATASET_FILES:
    path = Path(name)
    print(f'{name}: {"found" if path.exists() else "missing"}')
""",
    8: """import h5py
import numpy as np

def load_hdf5_sample(path, max_samples=15000):
    with h5py.File(path, 'r') as f:
        print(f'Keys: {list(f.keys())}')
        X = f['X'][:max_samples]
        if 'y' in f:
            y = f['y'][:max_samples]
        else:
            y = np.zeros(len(X), dtype=np.int64)
    return X, y

# Files are expected in the current working directory.
X_photon, y_photon = load_hdf5_sample('SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5')
X_electron, y_electron = load_hdf5_sample('SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5')

n_channels = min(X_photon.shape[-1], 2)
X_photon = X_photon[:, :, :, :n_channels]
X_electron = X_electron[:, :, :, :n_channels]

N = min(len(X_photon), len(X_electron))
X_photon = X_photon[:N]
X_electron = X_electron[:N]
labels_photon = np.zeros(N, dtype=np.int64)
labels_electron = np.ones(N, dtype=np.int64)

X_all = np.concatenate([X_photon, X_electron], axis=0)
y_all = np.concatenate([labels_photon, labels_electron], axis=0)

print(f'✅ X_all: {X_all.shape}, y_all: {y_all.shape}')
print(f'Class balance: {np.bincount(y_all)}')
""",
    11: """import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Convert to (N, C, H, W) format for PyTorch
X_all_t = np.transpose(X_all, (0, 3, 1, 2)).astype(np.float32)

# Normalise each channel independently
for c in range(X_all_t.shape[1]):
    ch = X_all_t[:, c, :, :]
    mean, std = float(ch.mean()), float(ch.std()) + 1e-8
    X_all_t[:, c, :, :] = (ch - mean) / std
    print(f'Channel {c} (mean={mean:.4f}, std={std:.4f})')

# Train / Val / Test split 80 / 10 / 10
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all_t, y_all, test_size=0.20, random_state=42, stratify=y_all
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f'\\nTrain: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}  |  Test: {X_test.shape[0]:,}')

def to_tensor(X, y):
    return TensorDataset(torch.tensor(X), torch.tensor(y))

train_ds = to_tensor(X_train, y_train)
val_ds = to_tensor(X_val, y_val)
test_ds = to_tensor(X_test, y_test)

cpu_workers = 0 if device.type == 'cpu' else 2
pin_memory = device.type == 'cuda'
batch_size = 128 if device.type == 'cpu' else 512

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=cpu_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=cpu_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=cpu_workers, pin_memory=pin_memory)

print('✅ DataLoaders ready!')
""",
    15: """criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

EPOCHS = 25 if device.type == 'cuda' else 3
history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
best_auc = 0.0

def evaluate(loader):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().to(device)
            logits = model(X_batch)
            total_loss += criterion(logits, y_batch).item() * len(y_batch)
            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    probs = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, probs)
    return total_loss / len(loader.dataset), auc, probs, labels

print(f'Training ResNet-15 for {EPOCHS} epochs...\\n')
print(f'{"Epoch":>5} | {"Train Loss":>10} | {"Val Loss":>8} | {"Val AUC":>8}')
print('-' * 45)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.float().to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y_batch)

    train_loss /= len(train_loader.dataset)
    val_loss, val_auc, _, _ = evaluate(val_loader)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(val_auc)

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), 'best_model_task1.pth')

    print(f'{epoch:>5} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_auc:>8.4f} {" ✅ best" if val_auc == best_auc else ""}')

print(f'\\nBest Validation AUC: {best_auc:.4f}')
""",
    17: """# Load best model and evaluate on test set
model.load_state_dict(torch.load('best_model_task1.pth', map_location=device))
test_loss, test_auc, test_probs, test_labels = evaluate(test_loader)
test_preds = (test_probs > 0.5).astype(int)
test_acc = accuracy_score(test_labels, test_preds)

print('=' * 40)
print('       TEST SET RESULTS')
print('=' * 40)
print(f'  AUC Score  : {test_auc:.4f}')
print(f'  Accuracy   : {test_acc:.4f}')
print(f'  Test Loss  : {test_loss:.4f}')
print('=' * 40)
""",
}

task2_replacements = {
    3: """# Install dependencies when needed
import importlib.util
import subprocess
import sys

required = ['h5py', 'sklearn', 'matplotlib', 'seaborn']
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]

if missing:
    print(f'Installing missing packages: {missing}')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
else:
    print('Dependencies already available.')
""",
    6: """import os
import shutil
import subprocess
from pathlib import Path
import numpy as np

def _is_valid_npz(path):
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path, allow_pickle=False)
        return 'X' in d and 'y' in d
    except Exception:
        return False

print('Attempting to download dataset...')
if not _is_valid_npz('QG_jets.npz'):
    wget = shutil.which('wget')
    if wget:
        try:
            subprocess.check_call([
                wget, '-q', '--no-check-certificate',
                'https://cernbox.cern.ch/index.php/s/hqZ8zE7oxyPjvsL/download',
                '-O', 'QG_jets.npz'
            ])
        except Exception as exc:
            print(f'Download failed: {exc}')

if _is_valid_npz('QG_jets.npz'):
    print('Dataset downloaded and validated successfully!')
else:
    print('Download failed. Will generate synthetic data in next cell.')
    path = Path('QG_jets.npz')
    if path.exists():
        path.unlink()
""",
    7: """import os
import numpy as np

N_PER_CLASS_GPU = 10000
N_PER_CLASS_CPU = 400

if not os.path.exists('QG_jets.npz'):
    print('Real dataset not found. Generating synthetic data...')

    try:
        import torch
        on_gpu = torch.cuda.is_available()
    except ImportError:
        on_gpu = False

    N_per_class = N_PER_CLASS_GPU if on_gpu else N_PER_CLASS_CPU
    print(f'{"GPU" if on_gpu else "CPU"} detected - generating {N_per_class:,} samples/class')
    print(f'Estimated RAM: ~{N_per_class * 2 * 125 * 125 * 3 * 4 / 1e9:.2f} GB')

    def make_jet_image(N, sigma, n_particles_mean):
        imgs = np.zeros((N, 125, 125, 3), dtype=np.float32)
        center = 62
        for i in range(N):
            if i % 250 == 0:
                print(f'  Generating image {i}/{N}...')
            n_hits = np.random.poisson(n_particles_mean)
            eta = np.random.normal(center, sigma, n_hits).astype(int)
            phi = np.random.normal(center, sigma, n_hits).astype(int)
            energy = np.abs(np.random.exponential(1.0, n_hits))
            mask = (eta >= 0) & (eta < 125) & (phi >= 0) & (phi < 125)
            np.add.at(imgs[i, :, :, 0], (eta[mask], phi[mask]), energy[mask])
            np.add.at(imgs[i, :, :, 1], (eta[mask], phi[mask]), energy[mask] * 0.3)
            np.add.at(imgs[i, :, :, 2], (eta[mask], phi[mask]), 1.0)
        return imgs

    X_gluon = make_jet_image(N_per_class, sigma=8, n_particles_mean=30)
    X_quark = make_jet_image(N_per_class, sigma=5, n_particles_mean=20)

    X_all = np.concatenate([X_gluon, X_quark], axis=0)
    y_all = np.concatenate([np.zeros(N_per_class), np.ones(N_per_class)]).astype(np.int64)
    del X_gluon, X_quark

    np.savez_compressed('QG_jets.npz', X=X_all, y=y_all)
    print(f'Saved: X={X_all.shape}, y={y_all.shape}')
else:
    print('Dataset found!')
""",
}


def replace_task2_training_cells(nb_path: Path) -> None:
    data = json.loads(nb_path.read_text(encoding="utf-8"))

    for idx, cell in enumerate(data["cells"]):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))

        if "EPOCHS   = 5" in source or "EPOCHS   = 30" in source or "EPOCHS = 30" in source:
            source = source.replace("EPOCHS   = 5", "EPOCHS   = 2 if device.type == 'cpu' else 5")
            source = source.replace("EPOCHS   = 30", "EPOCHS   = 2 if device.type == 'cpu' else 30")
            source = source.replace("EPOCHS = 30", "EPOCHS = 2 if device.type == 'cpu' else 30")
            source = source.replace("N = 2000  # small enough to not crash RAM", "N = 300  # CPU-safe fallback size")
            source = source.replace("N = 2000", "N = 300")
            source = source.replace(
                "return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())",
                "return DataLoader(ds, batch_size=bs if device.type == 'cuda' else min(bs, 64), shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())",
            )
            source = source.replace(
                "return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=0)",
                "return DataLoader(ds, batch_size=64 if device.type == 'cuda' else 32, shuffle=shuffle, num_workers=0)",
            )
            cell["source"] = source.splitlines(keepends=True)

    nb_path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


replace_cell_source(ROOT / "modified_Task1_Electron_Photon_Classification.ipynb", task1_replacements)
replace_cell_source(ROOT / "Task2_sparse_neural_network.ipynb", task2_replacements)
replace_task2_training_cells(ROOT / "Task2_sparse_neural_network.ipynb")
