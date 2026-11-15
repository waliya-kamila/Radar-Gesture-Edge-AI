"""Create host-test vectors.
 (1) inference: compact windows of participant 8 + TensorFlow Lite reference outputs
 (2) pre-processing: random raw frames + the Python (scipy) compact output
usage: python make_host_vectors.py demo_dir out_dir
"""
import sys, json, numpy as np
from scipy.signal import hilbert
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
demo, out = sys.argv[1], sys.argv[2]
d = np.load(f'{demo}/demo_data.npz'); norm = json.load(open(f'{demo}/norm.json'))['8']
idx = np.flatnonzero(d['subject'] == 8)
X = d['X'][idx].astype(np.float32); Y = d['y'][idx].astype(np.int32)
it = Interpreter(model_path=f'{demo}/models/subject8_int8.tflite'); it.allocate_tensors()
inp, outd = it.get_input_details()[0], it.get_output_details()[0]
sc, zp = inp['quantization']; ref = []
for x in X:
    z = x / (x.std() + 1e-8)
    z = (z - np.array(norm['mean'], np.float32)) / np.array(norm['std'], np.float32)
    it.set_tensor(inp['index'], np.clip(np.round(z / sc + zp), -128, 127).astype(np.int8)[None]); it.invoke()
    ref.append(it.get_tensor(outd['index'])[0])
with open(f'{out}/infer_vectors.bin', 'wb') as f:
    np.array([len(X)], np.int32).tofile(f); X.tofile(f); np.array(ref, np.int8).tofile(f); Y.tofile(f)
print('inference vectors:', len(X), '| TFLite acc', (np.array(ref).argmax(1) == Y).mean())

def make_compact(batch, alpha=0.9):                  # copy of radar_utils.make_compact
    out_ = np.empty_like(batch); bg = batch[:, 0].copy()
    for n in range(batch.shape[1]):
        bg = alpha * bg + (1 - alpha) * batch[:, n]; out_[:, n] = batch[:, n] - bg
    x = np.abs(hilbert(out_, axis=2)).astype(np.float32)[:, :, 34:162, :]
    B, T, R, C = x.shape
    return x.reshape(B, T, R // 4, 4, C).mean(axis=3).astype(np.float32)
rng = np.random.default_rng(0)
raw = (rng.normal(0, 1, (3, 90, 189, 3)) + 5 * np.sin(np.arange(189) / 7)[None, None, :, None]).astype(np.float32)
with open(f'{out}/prep_vectors.bin', 'wb') as f:
    np.array([len(raw)], np.int32).tofile(f); raw.tofile(f); make_compact(raw).tofile(f)
print('pre-processing vectors:', len(raw))
