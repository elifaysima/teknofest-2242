#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTLE HIZLI SON – SVM, RF, VQC (6 qubit, reps=1, maxiter=100)
Toplam süre ~5‑10 dakika.
"""

import numpy as np
import os
from scipy.signal import butter, filtfilt, hilbert
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
import mne
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ====================== AYARLAR ======================
BASE_DIRS = {
    "epilepsy":  "C:/Users/Lenovo/Desktop/ds003029",
    "control_BRAIN2": "C:/Users/Lenovo/Desktop/ds004993",
    "control_amygdala": "C:/Users/Lenovo/Desktop/ds003374"
}
TARGET_CHANNELS = 25
MIN_DURATION_SEC = 60
WINDOWS = [5, 10, 15, 20, 25]
WINDOW_DUR = 15
ELECTRODE_LAYOUT_FILE = "electrode_layout.xlsx"
# =====================================================

# ---------- Yardımcı fonksiyonlar ----------
def bandpass_filter(data, low=1, high=40, fs=1000, order=4):
    nyq = 0.5 * fs
    lowcut = low / nyq; highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, data, axis=1)

def compute_plv_matrix(data, fs=1000):
    filtered = bandpass_filter(data, low=1, high=40, fs=fs)
    analytic = hilbert(filtered)
    phase = np.angle(analytic)
    n_ch = phase.shape[0]
    plv = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i+1, n_ch):
            delta = phase[i] - phase[j]
            plv[i, j] = np.abs(np.mean(np.exp(1j * delta)))
            plv[j, i] = plv[i, j]
    np.fill_diagonal(plv, 0)
    return np.tanh(plv * 2 - 1)

def get_channels_from_layout(excel_path, max_channels, keywords=['hippocampus','amygdala']):
    df = pd.read_excel(excel_path)
    chan_col = None; region_col = None
    for c in df.columns:
        if 'channel' in c.lower(): chan_col = c
        if 'region' in c.lower() or 'label' in c.lower(): region_col = c
    if chan_col is None or region_col is None:
        raise ValueError("Excel'de Channel ve Region sütunu bulunamadı.")
    mask = df[region_col].astype(str).str.lower().str.contains('|'.join(keywords))
    selected = df.loc[mask, chan_col].unique().tolist()
    if len(selected) > max_channels:
        selected = selected[:max_channels]
    return selected

def load_all_runs(ieeg_dir):
    valid_exts = ['.edf', '.vhdr', '.set']
    file_list = []
    for f in sorted(os.listdir(ieeg_dir)):
        if any(f.endswith(ext) for ext in valid_exts):
            file_list.append(os.path.join(ieeg_dir, f))
    if not file_list:
        raise FileNotFoundError(f"{ieeg_dir} içinde desteklenen dosya bulunamadı.")
    def read_raw(path):
        if path.endswith('.edf'):
            return mne.io.read_raw_edf(path, preload=True)
        elif path.endswith('.vhdr'):
            return mne.io.read_raw_brainvision(path, preload=True)
        elif path.endswith('.set'):
            return mne.io.read_raw_eeglab(path, preload=True)
    raw = read_raw(file_list[0])
    for f in file_list[1:]:
        raw.append(read_raw(f))
    return raw

def extract_plv_features(W):
    n = W.shape[0]
    feats = [
        np.mean(W),
        np.std(W),
        np.sum(np.abs(W) > 0.5) / (n*(n-1)),
        np.mean(W[:10, :10]) if n>=10 else 0,
        np.mean(W[-5:, -5:]) if n>=5 else 0,
        np.mean(W[:10, -5:]) if n>=15 else 0,
        np.mean(np.abs(W[np.triu_indices(n, k=1)] - W[np.tril_indices(n, k=-1)]))
    ]
    return np.array(feats)

def build_ising_hamiltonian(W, bias=None):
    n = W.shape[0]
    if bias is None: bias = np.zeros(n)
    pauli_list, coeff_list = [], []
    for i in range(n):
        for j in range(i+1, n):
            if abs(W[i,j]) > 1e-6:
                label = ['I']*n; label[i]='Z'; label[j]='Z'
                pauli_list.append(''.join(label))
                coeff_list.append(-W[i,j])
    for i in range(n):
        if abs(bias[i]) > 1e-6:
            label = ['I']*n; label[i]='Z'
            pauli_list.append(''.join(label))
            coeff_list.append(-bias[i])
    if not pauli_list:
        return SparsePauliOp(['I'*n], coeffs=[0.0])
    return SparsePauliOp(pauli_list, coeffs=np.array(coeff_list))

def run_vqe(H, n_qubits=5):
    n_qubits = min(n_qubits, H.num_qubits)
    reduced_ops, coeffs = [], []
    for op, coeff in zip(H.paulis, H.coeffs):
        label = str(op)[:n_qubits]
        if len(label) < n_qubits: label += 'I'*(n_qubits-len(label))
        reduced_ops.append(label)
        coeffs.append(coeff)
    H_red = SparsePauliOp(reduced_ops, coeffs=np.array(coeffs))
    ansatz = TwoLocal(n_qubits, 'ry', 'cz', reps=2, entanglement='linear')
    opt = COBYLA(maxiter=100)
    vqe = VQE(StatevectorEstimator(), ansatz, opt)
    res = vqe.compute_minimum_eigenvalue(H_red)
    qc = ansatz.assign_parameters(res.optimal_point)
    qc.measure_all()
    sampler = StatevectorSampler(seed=42)
    job = sampler.run([qc], shots=4096)
    counts = job.result()[0].data.meas.get_counts()
    max_bit = max(counts, key=counts.get)
    return res.optimal_value, max_bit, counts

def extract_features_with_vqe(W):
    classic = extract_plv_features(W)
    H = build_ising_hamiltonian(W)
    energy, _, _ = run_vqe(H, n_qubits=5)
    return np.append(classic, energy)

def process_dataset(base_dir, label, excel_path, target_ch, min_dur, windows, win_dur):
    all_subjects = sorted([d for d in os.listdir(base_dir) if d.startswith('sub-')])
    X_list, y_list = [], []
    for sub in all_subjects:
        sub_path = os.path.join(base_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        # ieeg klasörünü bul
        ieeg_dir = None
        for root, dirs, files in os.walk(sub_path):
            if 'ieeg' in dirs:
                ieeg_dir = os.path.join(root, 'ieeg')
                break
        if ieeg_dir is None:
            ses_dirs = [d for d in os.listdir(sub_path) if d.startswith('ses-')]
            for ses in ses_dirs:
                possible = os.path.join(sub_path, ses, 'ieeg')
                if os.path.isdir(possible):
                    ieeg_dir = possible
                    break
        if ieeg_dir is None:
            ieeg_dir = sub_path
        try:
            raw = load_all_runs(ieeg_dir)
        except:
            continue
        total_dur = raw.times[-1]
        n_chan = raw.info['nchan']
        if total_dur < min_dur or n_chan < target_ch:
            continue
        # kanal seçimi
        if excel_path and os.path.exists(excel_path):
            try:
                chosen = get_channels_from_layout(excel_path, target_ch)
            except:
                chosen = raw.ch_names[:target_ch]
        else:
            chosen = raw.ch_names[:target_ch]
        common = [ch for ch in chosen if ch in raw.ch_names]
        if len(common) < target_ch:
            for ch in raw.ch_names:
                if ch not in common:
                    common.append(ch)
                if len(common) >= target_ch:
                    break
        raw.pick_channels(common[:target_ch])
        fs = raw.info['sfreq']
        data = raw.get_data()
        data_mean = np.mean(data, axis=1, keepdims=True)
        data_std = np.std(data, axis=1, keepdims=True) + 1e-8
        data_norm = (data - data_mean) / data_std
        for start_sec in windows:
            start_idx = int(start_sec * fs)
            if start_idx + win_dur * fs > data_norm.shape[1]:
                continue
            segment = data_norm[:, start_idx : start_idx + int(win_dur * fs)]
            W = compute_plv_matrix(segment, fs=fs)
            feat = extract_features_with_vqe(W)
            X_list.append(feat)
            y_list.append(label)
    return X_list, y_list

# ====================== ANA AKIŞ ======================
if __name__ == "__main__":
    print("🚀 HIZLI MTLE Kuantum Sistemi")
    excel_path = os.path.join(BASE_DIRS["epilepsy"], ELECTRODE_LAYOUT_FILE)
    if not os.path.exists(excel_path):
        excel_path = None

    X_all, y_all = [], []
    # 1. Epilepsy
    print("Epilepsi verileri işleniyor...")
    X1, y1 = process_dataset(BASE_DIRS["epilepsy"], 1, excel_path, TARGET_CHANNELS, MIN_DURATION_SEC, WINDOWS, WINDOW_DUR)
    X_all.extend(X1); y_all.extend(y1)
    print(f"   {len(X1)} örnek eklendi.")
    # 2. B(RAIN)2
    print("B(RAIN)2 kontrol verileri işleniyor...")
    X2, y2 = process_dataset(BASE_DIRS["control_BRAIN2"], 0, None, TARGET_CHANNELS, MIN_DURATION_SEC, WINDOWS, WINDOW_DUR)
    X_all.extend(X2); y_all.extend(y2)
    print(f"   {len(X2)} örnek eklendi.")
    # 3. Amigdala kontrol
    print("Amigdala kontrol verileri işleniyor...")
    X3, y3 = process_dataset(BASE_DIRS["control_amygdala"], 0, None, TARGET_CHANNELS, MIN_DURATION_SEC, WINDOWS, WINDOW_DUR)
    X_all.extend(X3); y_all.extend(y3)
    print(f"   {len(X3)} örnek eklendi.")

    if len(X_all) < 10:
        raise RuntimeError("Yeterli örnek oluşmadı!")
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    print(f"\nToplam {len(X_all)} örnek (0: kontrol, 1: hasta)")

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sonuc = {'SVM': [], 'RF': [], 'VQC': []}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_all, y_all)):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        # SVM
        svm = SVC(kernel='rbf').fit(X_tr, y_tr)
        sonuc['SVM'].append(accuracy_score(y_te, svm.predict(X_te)))

        # RF
        rf = RandomForestClassifier(n_estimators=100).fit(X_tr, y_tr)
        sonuc['RF'].append(accuracy_score(y_te, rf.predict(X_te)))

        # VQC (6 qubit, hızlı)
        # Eski VQC bloğunu bul ve aşağıdakiyle değiştir
vqc = VQC(
    feature_map=PauliFeatureMap(
        feature_dimension=6,
        reps=2,                         # derinliği artırdık
        paulis=['X','Y','Z','ZZ']       # zenginleştirilmiş pauli seti
    ),
    ansatz=RealAmplitudes(
        num_qubits=6,
        reps=2,                         # daha derin ansatz
        entanglement='full'             # tam bağlantı
    ),
    sampler=StatevectorSampler(seed=42),
    optimizer=COBYLA(maxiter=150)       # iterasyon biraz artırıldı
)
vqc.fit(X_tr[:, :6], y_tr)
sonuc['VQC'].append(accuracy_score(y_te, vqc.predict(X_te[:, :6])))

print("\n📊 5‑KATLI CV SONUÇLARI")
for name, accs in sonuc.items():
        print(f"{name}: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
print("✅ Tamamlandı.")