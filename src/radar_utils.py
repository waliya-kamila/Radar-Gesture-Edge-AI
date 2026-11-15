"""Shared helpers for the Radar Gesture Edge-AI project."""
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path('/content/drive/MyDrive/RadarGestureEdgeAI')
LOCAL = Path('/content/uwb_processed')
DATA_FILES = ['X_recordings.npy', 'recording_metadata.csv', 'y_labels.npy']

CHANNELS = ['Left', 'Top', 'Right']
FRAMES, BINS, NUM_CLASSES = 90, 189, 12
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_data():
    LOCAL.mkdir(exist_ok=True)
    for f in DATA_FILES:
        if not (LOCAL / f).exists():
            shutil.copy2(PROJECT / 'processed' / f, LOCAL / f)
    X = np.load(LOCAL / 'X_recordings.npy', mmap_mode='r')
    y = np.load(LOCAL / 'y_labels.npy')
    meta = pd.read_csv(LOCAL / 'recording_metadata.csv')
    return X, y, meta


def loso_folds(meta):
    subj = meta['participant_id'].to_numpy()
    keep = ~meta['is_duplicate'].to_numpy()
    for k, test_s in enumerate(SUBJECTS):
        val_s = SUBJECTS[(k + 1) % len(SUBJECTS)]
        train_s = [s for s in SUBJECTS if s not in (test_s, val_s)]
        yield {
            'fold': k + 1,
            'test_subject': test_s,
            'val_subject': val_s,
            'train_subjects': train_s,
            'train_idx': np.flatnonzero(keep & np.isin(subj, train_s)),
            'val_idx': np.flatnonzero(keep & (subj == val_s)),
            'test_idx': np.flatnonzero(keep & (subj == test_s)),
        }


def channel_stats(X, idx, chunk=256):
    total = np.zeros(len(CHANNELS))
    total_sq = np.zeros(len(CHANNELS))
    count = 0
    for i in range(0, len(idx), chunk):
        batch = np.asarray(X[idx[i:i + chunk]], dtype=np.float64)
        total += batch.sum(axis=(0, 1, 2))
        total_sq += (batch ** 2).sum(axis=(0, 1, 2))
        count += batch.shape[0] * batch.shape[1] * batch.shape[2]
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)
    return mean.astype(np.float32), std.astype(np.float32)


def evaluate(y_true, y_pred):
    """Macro-F1 over classes present in this test subject.
    (Subject 2 has no valid G11/G12 after duplicate removal.)"""
    from sklearn.metrics import accuracy_score, f1_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    present = np.unique(y_true)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, labels=present,
                             average='macro', zero_division=0),
    }


def save_results(df, name):
    out = PROJECT / 'results' / f'{name}.csv'
    df.to_csv(out, index=False)
    print('Saved:', out)


# ================= Phase 2: Signal processing =================

def clutter_mean_batch(batch):
    """batch: (B, frames, bins, channels). Subtract each bin's mean over time."""
    return batch - batch.mean(axis=1, keepdims=True)


def clutter_loopback_batch(batch, alpha=0.9):
    """Causal running-background subtraction, vectorised over the batch."""
    out = np.empty_like(batch)
    background = batch[:, 0].copy()
    for n in range(batch.shape[1]):
        background = alpha * background + (1 - alpha) * batch[:, n]
        out[:, n] = batch[:, n] - background
    return out


def block_features(batch):
    """(B, 90, 189, 3) -> (B, 729): mean, std, abs-mean of 9x9 time-range blocks."""
    blocks = batch.reshape(-1, 9, 10, 9, 21, 3)
    return np.concatenate([
        blocks.mean(axis=(2, 4)).reshape(len(batch), -1),
        blocks.std(axis=(2, 4)).reshape(len(batch), -1),
        np.abs(blocks).mean(axis=(2, 4)).reshape(len(batch), -1),
    ], axis=1).astype(np.float32)


def compute_features(X, transform=None, chunk=256):
    """Apply optional preprocessing per batch, then extract block features."""
    F = np.zeros((len(X), 729), dtype=np.float32)
    for s in range(0, len(X), chunk):
        batch = np.asarray(X[s:s + chunk], dtype=np.float32)
        if transform is not None:
            batch = transform(batch)
        F[s:s + chunk] = block_features(batch)
    return F


def run_loso_baselines(F, y, meta, tag, verbose=True):
    """LogReg + RandomForest on all 8 LOSO folds. Saves results to Drive."""
    import time
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    set_seed()
    rows, preds = [], []
    for f in loso_folds(meta):
        tr, te = f['train_idx'], f['test_idx']
        models = {
            'LogReg': make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_features='sqrt', min_samples_leaf=2,
                random_state=SEED, n_jobs=-1),
        }
        for name, model in models.items():
            t0 = time.time()
            model.fit(F[tr], y[tr])
            p = model.predict(F[te])
            m = evaluate(y[te], p)
            rows.append({'model': name, 'fold': f['fold'],
                         'test_subject': f['test_subject'], **m})
            preds.append(pd.DataFrame({'model': name, 'fold': f['fold'], 'idx': te,
                                       'y_true': y[te], 'y_pred': p}))
            if verbose:
                print(f"[{tag}] Fold {f['fold']} (S{f['test_subject']}) | {name:12s} | "
                      f"acc={m['accuracy']:.3f} | F1={m['macro_f1']:.3f} | "
                      f"{time.time() - t0:.0f}s")
    results = pd.DataFrame(rows)
    save_results(results, f'loso_baselines_{tag}')
    save_results(pd.concat(preds), f'loso_baselines_{tag}_predictions')
    return results


# ================= Step 2.4: Envelope + compact representation =================
RANGE_CROP = (34, 162)   # 128 bins containing ~95% of motion energy (all subjects)
RANGE_POOL = 4           # 128 -> 32 range cells


def envelope_batch(batch):
    """RF signal -> envelope along the range axis."""
    from scipy.signal import hilbert
    return np.abs(hilbert(batch, axis=2)).astype(np.float32)


def make_compact(batch, alpha=0.9, crop=RANGE_CROP, pool=RANGE_POOL):
    """Raw RF (B, 90, 189, 3) -> compact envelope (B, 90, 128/pool, 3)."""
    x = clutter_loopback_batch(batch, alpha=alpha)
    x = envelope_batch(x)
    x = x[:, :, crop[0]:crop[1], :]
    B, T, R, C = x.shape
    return x.reshape(B, T, R // pool, pool, C).mean(axis=3).astype(np.float32)


def build_compact_dataset(X, out_path, chunk=256, **kwargs):
    """Apply make_compact to the whole dataset, writing to a .npy file."""
    first = make_compact(np.asarray(X[:1], dtype=np.float32), **kwargs)
    Xc = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32,
                                   shape=(len(X),) + first.shape[1:])
    for s in range(0, len(X), chunk):
        Xc[s:s + chunk] = make_compact(np.asarray(X[s:s + chunk], dtype=np.float32), **kwargs)
    Xc.flush()
    return Xc


def compact_block_features(batch, t_blocks=9, r_blocks=8):
    """(B, 90, 32, 3) -> (B, 648): mean, std, max of 9x8 time-range blocks."""
    B, T, R, C = batch.shape
    blocks = batch.reshape(B, t_blocks, T // t_blocks, r_blocks, R // r_blocks, C)
    return np.concatenate([
        blocks.mean(axis=(2, 4)).reshape(B, -1),
        blocks.std(axis=(2, 4)).reshape(B, -1),
        blocks.max(axis=(2, 4)).reshape(B, -1),
    ], axis=1).astype(np.float32)


# ================= Phase 3: Tiny CNN (final training recipe) =================
import time as _time
import torch
import torch.nn as nn
import torch.nn.functional as _F

COMPACT_FILES = ['X_compact.npy', 'y_labels.npy', 'recording_metadata.csv']


def load_compact():
    """Load compact dataset (90 x 32 x 3) into RAM. Copies from Drive once per session."""
    LOCAL.mkdir(exist_ok=True)
    for f in COMPACT_FILES:
        if not (LOCAL / f).exists():
            shutil.copy2(PROJECT / 'processed' / f, LOCAL / f)
    Xc = np.load(LOCAL / 'X_compact.npy')
    y = np.load(LOCAL / 'y_labels.npy')
    meta = pd.read_csv(LOCAL / 'recording_metadata.csv')
    return Xc, y, meta


class TinyRadarCNN(nn.Module):
    """Edge-sized CNN for compact radar input (in_ch x 90 x 32)."""
    def __init__(self, in_ch=3, num_classes=NUM_CLASSES, widths=(16, 32, 64)):
        super().__init__()
        w1, w2, w3 = widths

        def block(cin, cout, pool):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                      nn.BatchNorm2d(cout), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return layers

        self.features = nn.Sequential(
            *block(in_ch, w1, True), *block(w1, w2, True), *block(w2, w3, False),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(w3, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def prepare_input(Xc, idx, mode='persample', channels=(0, 1, 2)):
    """Per-recording input transform (no cross-subject statistics)."""
    x = Xc[idx][..., list(channels)].astype(np.float32)
    if mode == 'log':
        x = np.log(x + 1e-4)
    elif mode == 'persample':
        x = x / (x.std(axis=(1, 2, 3), keepdims=True) + 1e-8)
    return x


def augment(x, max_t=0, max_r=0, noise=0.0):
    """Random time/range shift (edge-replicate padding) + Gaussian noise."""
    B, C, T, R = x.shape
    if max_t or max_r:
        xp = _F.pad(x, (max_r, max_r, max_t, max_t), mode='replicate')
        ts = torch.randint(0, 2 * max_t + 1, (B,)).tolist()
        rs = torch.randint(0, 2 * max_r + 1, (B,)).tolist()
        x = torch.stack([xp[i, :, ts[i]:ts[i] + T, rs[i]:rs[i] + R] for i in range(B)])
    if noise > 0:
        x = x + noise * torch.randn_like(x)
    return x


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    return torch.cat([model(X[s:s + bs]).argmax(1)
                      for s in range(0, len(X), bs)]).cpu().numpy()


def train_cnn_fold(Xc, y, fold, device, mode='persample', channels=(0, 1, 2),
                   aug=None, widths=(16, 32, 64), epochs=60, patience=15,
                   min_epochs=20, batch_size=64, lr=1e-3, seed=None):
    """Final training recipe from Phase 3."""
    aug = aug or {}
    seed = SEED if seed is None else seed
    set_seed(seed + fold['fold'])
    tr, va, te = fold['train_idx'], fold['val_idx'], fold['test_idx']

    xtr = prepare_input(Xc, tr, mode, channels)
    mean = xtr.mean(axis=(0, 1, 2))
    std = xtr.std(axis=(0, 1, 2))

    def to_t(x, idx):
        x = np.transpose((x - mean) / std, (0, 3, 1, 2))
        return (torch.tensor(x, dtype=torch.float32, device=device),
                torch.tensor(y[idx], device=device))

    Xtr, ytr = to_t(xtr, tr)
    Xva, yva = to_t(prepare_input(Xc, va, mode, channels), va)
    Xte, _ = to_t(prepare_input(Xc, te, mode, channels), te)
    del xtr

    model = TinyRadarCNN(in_ch=len(channels), widths=widths).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_ep, best_state, bad = -1.0, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(augment(Xtr[b], **aug)), ytr[b]).backward()
            opt.step()
        sched.step()

        val_f1 = evaluate(yva.cpu().numpy(), predict(model, Xva))['macro_f1']
        if val_f1 > best_f1:
            best_f1, best_ep, bad = val_f1, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience and ep >= min_epochs:
                break

    model.load_state_dict(best_state)
    return {'model': model, 'y_pred': predict(model, Xte), 'val_f1': best_f1,
            'best_epoch': best_ep, 'last_epoch': ep, 'mean': mean, 'std': std}


def run_cnn_loso(Xc, y, meta, device, tag, save_models=False, verbose=True, **kwargs):
    """Train/evaluate the CNN on all 8 LOSO folds. Saves results (and optionally models)."""
    rows, preds = [], []
    t0 = _time.time()
    for fold in loso_folds(meta):
        r = train_cnn_fold(Xc, y, fold, device, **kwargs)
        te = fold['test_idx']
        m = evaluate(y[te], r['y_pred'])
        rows.append({'model': tag, 'fold': fold['fold'],
                     'test_subject': fold['test_subject'], **m,
                     'best_epoch': r['best_epoch']})
        preds.append(pd.DataFrame({'model': tag, 'fold': fold['fold'], 'idx': te,
                                   'y_true': y[te], 'y_pred': r['y_pred']}))
        if save_models:
            torch.save({'state_dict': r['model'].state_dict(), 'mean': r['mean'],
                        'std': r['std'], 'config': kwargs},
                       PROJECT / 'models' / f"{tag}_fold{fold['fold']}.pt")
        if verbose:
            print(f"[{tag}] Fold {fold['fold']} (S{fold['test_subject']}) | "
                  f"best ep={r['best_epoch']:2d}/{r['last_epoch']:2d} | "
                  f"TEST F1={m['macro_f1']:.3f}")
    df = pd.DataFrame(rows)
    save_results(df, f'loso_{tag}')
    save_results(pd.concat(preds), f'loso_{tag}_predictions')
    print(f"[{tag}] mean F1 = {df.macro_f1.mean() * 100:.1f} | "
          f"worst = {df.macro_f1.min() * 100:.1f} | {(_time.time() - t0) / 60:.1f} min")
    return df


# ================= Phase 5: Edge metrics =================
def count_macs(model, in_ch=3, T=FRAMES, R=32):
    """Multiply-accumulate operations for one inference (Conv2d + Linear)."""
    macs = 0

    def conv_hook(m, inp, out):
        nonlocal macs
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs += out.numel() // out.shape[0] * k

    def lin_hook(m, inp, out):
        nonlocal macs
        macs += m.in_features * m.out_features

    hooks = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, in_ch, T, R))
    for h in hooks:
        h.remove()
    return macs


# ================= Phase 5.2: INT8 deployment (PyTorch -> TFLite) =================
def load_fold_model(tag, fold_k, widths=(16, 32, 64), in_ch=3):
    """Load a saved LOSO fold model + its input normalisation stats."""
    ck = torch.load(PROJECT / 'models' / f'{tag}_fold{fold_k}.pt',
                    map_location='cpu', weights_only=False)
    model = TinyRadarCNN(in_ch=in_ch, widths=widths)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, np.asarray(ck['mean'], np.float32), np.asarray(ck['std'], np.float32)


def normalize_input(Xc, idx, mean, std, channels=(0, 1, 2)):
    """Same input pipeline as training -> (N, 90, 32, C) float32 (NHWC)."""
    return ((prepare_input(Xc, idx, 'persample', channels) - mean) / std).astype(np.float32)


def fold_bn(model):
    """Fold every BatchNorm into the conv before it (standard MCU deployment step)."""
    layers = [m for m in model.modules()
              if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear))]
    convs = []
    with torch.no_grad():
        for conv, bn in zip(layers[0:-1:2], layers[1:-1:2]):
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w = conv.weight * s[:, None, None, None]
            b = bn.bias - bn.running_mean * s
            convs.append((w.permute(2, 3, 1, 0).numpy().copy(), b.numpy().copy()))  # OIHW -> HWIO
        lin = layers[-1]
        dense = (lin.weight.T.numpy().copy(), lin.bias.numpy().copy())
    return convs, dense


def build_tf_model(model):
    """Re-create the trained TinyRadarCNN as a TensorFlow graph (NHWC, BN folded)."""
    import tensorflow as tf
    convs, (wd, bd) = fold_bn(model)

    class TFRadarCNN(tf.Module):
        def __init__(self):
            super().__init__()
            self.w = [tf.Variable(w, trainable=False) for w, _ in convs]
            self.b = [tf.Variable(b, trainable=False) for _, b in convs]
            self.wd = tf.Variable(wd, trainable=False)
            self.bd = tf.Variable(bd, trainable=False)

        @tf.function
        def __call__(self, x):
            for i in range(len(self.w)):
                x = tf.nn.relu(tf.nn.conv2d(x, self.w[i], 1, 'SAME') + self.b[i])
                if i < len(self.w) - 1:                      # blocks 1, 2 have MaxPool
                    x = tf.nn.max_pool2d(x, 2, 2, 'VALID')
            x = tf.reduce_mean(x, axis=[1, 2])               # global average pool
            return tf.matmul(x, self.wd) + self.bd

    return TFRadarCNN()


def convert_tflite(tf_model, rep_x=None, out_path=None):
    """TFLite export. rep_x given -> full INT8 (weights + activations + I/O), else float32."""
    import tensorflow as tf
    spec = tf.TensorSpec([1, *rep_x.shape[1:]] if rep_x is not None
                         else [1, FRAMES, 32, tf_model.w[0].shape[2]], tf.float32)
    cf = tf_model.__call__.get_concrete_function(spec)
    conv = tf.lite.TFLiteConverter.from_concrete_functions([cf], tf_model)
    if rep_x is not None:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([rep_x[i:i + 1]] for i in range(len(rep_x)))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    tfl = conv.convert()
    if out_path is not None:
        Path(out_path).write_bytes(tfl)
    return tfl


def tflite_predict(tfl, x):
    """Run a .tflite model sample-by-sample (like a microcontroller would)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_content=tfl)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds = np.empty(len(x), dtype=np.int64)
    for i in range(len(x)):
        xi = x[i:i + 1]
        if inp['dtype'] == np.int8:
            s, z = inp['quantization']
            xi = np.clip(np.round(xi / s + z), -128, 127).astype(np.int8)
        it.set_tensor(inp['index'], xi)
        it.invoke()
        preds[i] = it.get_tensor(out['index'])[0].argmax()
    return preds


# ================= Phase 6: Privacy (identity leakage + adversarial de-identification) =================
from sklearn.preprocessing import StandardScaler as _SS
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.ensemble import RandomForestClassifier as _RF
from sklearn.pipeline import make_pipeline as _pipe
from sklearn.metrics import accuracy_score as _acc


def within_subject_split(meta, val_frac=0.15, test_frac=0.25):
    """Chronological split INSIDE every (subject, gesture): early recordings = train,
    later = val/test. All 8 subjects appear in every split (needed to measure identity)."""
    m = meta[~meta['is_duplicate']].copy()
    m['_r'] = pd.to_numeric(m['recording_number'], errors='coerce')
    m = m.sort_values(['participant_id', 'label', '_r'], kind='stable')
    g = m.groupby(['participant_id', 'label'])
    frac = (g.cumcount() / g['label'].transform('size')).to_numpy()
    idx = m.index.to_numpy()
    return {'fold': 0,
            'train_idx': np.sort(idx[frac < 1 - val_frac - test_frac]),
            'val_idx': np.sort(idx[(frac >= 1 - val_frac - test_frac) & (frac < 1 - test_frac)]),
            'test_idx': np.sort(idx[frac >= 1 - test_frac])}


def identity_probe(F_tr, s_tr, F_te, s_te):
    """Post-hoc attacker: train a fresh classifier to predict WHO performed the gesture."""
    res = {}
    for name, clf in [('LR', _pipe(_SS(), _LR(max_iter=3000))),
                      ('RF', _RF(n_estimators=200, n_jobs=-1, random_state=SEED))]:
        clf.fit(F_tr, s_tr)
        res[name] = _acc(s_te, clf.predict(F_te))
    return res


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class AdvRadarCNN(nn.Module):
    """TinyRadarCNN + identity adversary.
    Only .net (the plain TinyRadarCNN) is deployed -> zero extra cost on the MCU."""
    def __init__(self, n_subjects, in_ch=3, widths=(16, 32, 64)):
        super().__init__()
        self.net = TinyRadarCNN(in_ch=in_ch, widths=widths)
        self.id_head = nn.Sequential(nn.Linear(widths[-1], 64), nn.ReLU(),
                                     nn.Linear(64, n_subjects))
        self.lam = 0.0
        self.mode = 'grl'      # 'grl' = gradient reversal, 'conf' = confusion (uniform) loss

    def forward(self, x):
        z = self.net.features(x).flatten(1)
        if self.mode == 'grl':
            return self.net.classifier(z), self.id_head(_GradReverse.apply(z, self.lam)), None
        frozen = {k: v.detach() for k, v in self.id_head.named_parameters()}
        conf = torch.func.functional_call(self.id_head, frozen, (z,))
        return self.net.classifier(z), self.id_head(z.detach()), conf


@torch.no_grad()
def embed(net, X, bs=512, level='embedding'):
    """What an attacker could see: 64-d embedding (GAP output) or 12-d output logits."""
    net.eval()
    f = (lambda b: net.features(b).flatten(1)) if level == 'embedding' else net
    return torch.cat([f(X[s:s + bs]) for s in range(0, len(X), bs)]).cpu().numpy()


def train_adv_fold(Xc, y, subj, fold, device, lam=1.0, widths=(16, 32, 64),
                   channels=(0, 1, 2), epochs=60, patience=15, min_epochs=20,
                   batch_size=64, lr=1e-3, seed=None, mode='grl'):
    """Gesture training with subject-adversarial de-identification.
    mode='grl': gradient reversal (DANN); mode='conf': push identity predictions to uniform.
    lam=0 -> normal training (adversary has no influence on the features).
    Model selection only from epoch >= min_epochs, when the adversary ramp is ~fully on."""
    seed = SEED if seed is None else seed
    set_seed(seed + fold['fold'])
    tr, va, te = fold['train_idx'], fold['val_idx'], fold['test_idx']
    _, s_tr = np.unique(subj[tr], return_inverse=True)

    xtr = prepare_input(Xc, tr, 'persample', channels)
    mean, std = xtr.mean(axis=(0, 1, 2)), xtr.std(axis=(0, 1, 2))
    to_t = lambda x: torch.tensor(np.transpose((x - mean) / std, (0, 3, 1, 2)),
                                  dtype=torch.float32, device=device)
    Xtr = to_t(xtr); del xtr
    Xva = to_t(prepare_input(Xc, va, 'persample', channels))
    Xte = to_t(prepare_input(Xc, te, 'persample', channels))
    ytr = torch.tensor(y[tr], device=device)
    str_ = torch.tensor(s_tr, device=device)

    model = AdvRadarCNN(int(s_tr.max()) + 1, in_ch=len(channels), widths=widths).to(device)
    model.mode = mode
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    best_f1, best_ep, best_state, bad = -1.0, 0, None, 0
    for ep in range(1, epochs + 1):
        p = (ep - 1) / epochs
        model.lam = lam * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)     # DANN ramp-up
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            if len(b) < 2:
                continue
            g_out, id_out, conf = model(Xtr[b])
            loss = ce(g_out, ytr[b]) + ce(id_out, str_[b])
            if conf is not None:
                loss = loss - model.lam * torch.log_softmax(conf, 1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep < min_epochs:
            continue
        val_f1 = evaluate(y[va], predict(model.net, Xva))['macro_f1']
        if val_f1 > best_f1:
            best_f1, best_ep, bad = val_f1, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    net = model.net.eval()
    return {'net': net, 'y_pred': predict(net, Xte), 'val_f1': best_f1,
            'best_epoch': best_ep, 'last_epoch': ep, 'mean': mean, 'std': std,
            'emb_train': embed(net, Xtr), 'emb_test': embed(net, Xte),
            'logit_train': embed(net, Xtr, level='logits'),
            'logit_test': embed(net, Xte, level='logits')}
