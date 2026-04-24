#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTLE iEEG Kuantum Sistemi - ÇOKLU HASTA, VQC İYİLEŞTİRMESİ, ÇAPRAZ DOĞRULAMA
"""

import numpy as np
import time, os, sys, glob
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
import mne
import warnings
warnings.filterwarnings('ignore')

# ====================== AYARLAR ======================
DATASET_DIR = "C:/Users/Lenovo/Desktop/ds003029"
TARGET_CHANNELS = 25
PREICTAL_DUR = 15
DURATION_SEC = 60
ELECTRODE_LAYOUT_FILE = "electrode_layout.xlsx"
# =====================================================

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
    print(f"   [KANAL SEÇİMİ] {len(selected)} kanal bulundu.")
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
    feats = []
    feats.append(np.mean(W))
    feats.append(np.std(W))
    feats.append(np.sum(np.abs(W) > 0.5) / (n*(n-1)))
    feats.append(np.mean(W[:10, :10]) if n>=10 else 0)
    feats.append(np.mean(W[-5:, -5:]) if n>=5 else 0)
    feats.append(np.mean(W[:10, -5:]) if n>=15 else 0)
    upper = W[np.triu_indices(n, k=1)]
    lower = W[np.tril_indices(n, k=-1)]
    feats.append(np.mean(np.abs(upper - lower)))
    return np.array(feats)

def process_single_subject(subject_path, electrode_excel, target_ch, dur_sec, preictal_sec):
    sub_id = os.path.basename(subject_path)
    print(f"\n{'='*50}\nHasta işleniyor: {sub_id}")

    ieeg_dir = None
    for root, dirs, files in os.walk(subject_path):
        if 'ieeg' in dirs:
            ieeg_dir = os.path.join(root, 'ieeg')
            break
    if ieeg_dir is None:
        ses_dirs = [d for d in os.listdir(subject_path) if d.startswith('ses-')]
        for ses in ses_dirs:
            possible = os.path.join(subject_path, ses, 'ieeg')
            if os.path.isdir(possible):
                ieeg_dir = possible
                break
    if ieeg_dir is None:
        ieeg_dir = subject_path
    print(f"   ieeg klasörü: {ieeg_dir}")

    try:
        raw = load_all_runs(ieeg_dir)
    except Exception as e:
        print(f"   ❌ Yükleme hatası: {e}")
        return None, None, None, None

    print(f"   Toplam kayıt: {raw.times[-1]:.1f} sn, {raw.info['nchan']} kanal")

    if electrode_excel and os.path.exists(electrode_excel):
        try:
            chosen = get_channels_from_layout(electrode_excel, target_ch)
        except Exception as e:
            print(f"   ⚠️  Elektrot seçimi başarısız: {e}, ilk 25 kanal alınıyor.")
            chosen = raw.ch_names[:target_ch]
    else:
        chosen = raw.ch_names[:target_ch]
    print(f"   Seçilen kanal sayısı: {len(chosen)}")

    common = [ch for ch in chosen if ch in raw.ch_names]
    if len(common) < target_ch:
        for ch in raw.ch_names:
            if ch not in common:
                common.append(ch)
            if len(common) >= target_ch:
                break
    raw.pick_channels(common[:target_ch])
    print(f"   Nihai kanal sayısı: {raw.info['nchan']}")

    fs = raw.info['sfreq']
    data = raw.get_data()
    n_samp = int(dur_sec * fs)
    if data.shape[1] > n_samp:
        data = data[:, :n_samp]

    n_pre = int(preictal_sec * fs)
    if n_pre < data.shape[1]:
        data_healthy = data[:, :n_pre]
        data_patho = data[:, n_pre:]
    else:
        half = data.shape[1] // 2
        data_healthy = data[:, :half]
        data_patho = data[:, half:]

    W_healthy = compute_plv_matrix(data_healthy, fs=fs)
    W_patho = compute_plv_matrix(data_patho, fs=fs)

    feat_h = extract_plv_features(W_healthy)
    feat_p = extract_plv_features(W_patho)

    return feat_h, feat_p, W_healthy, W_patho

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

if __name__ == "__main__":
    print("🚀 MTLE ÇOKLU HASTA ANALİZİ")
    all_subjects = sorted([d for d in os.listdir(DATASET_DIR) if d.startswith('sub-')])
    print(f"Toplam {len(all_subjects)} hasta bulundu: {all_subjects[:5]}...")

    excel_path = os.path.join(DATASET_DIR, ELECTRODE_LAYOUT_FILE)
    if not os.path.exists(excel_path):
        print("UYARI: electrode_layout.xlsx bulunamadı, ilk 25 kanal kullanılacak.")
        excel_path = None

    X_all = []
    y_all = []
    healthy_mats = []
    patho_mats = []

    for sub in all_subjects:
        sub_path = os.path.join(DATASET_DIR, sub)
        if not os.path.isdir(sub_path):
            continue
        result = process_single_subject(
            sub_path, excel_path, TARGET_CHANNELS, DURATION_SEC, PREICTAL_DUR
        )
        if result is None or result[0] is None:
            continue
        feat_h, feat_p, W_h, W_p = result
        X_all.append(feat_h)
        y_all.append(0)
        X_all.append(feat_p)
        y_all.append(1)
        healthy_mats.append(W_h)
        patho_mats.append(W_p)

    if len(X_all) < 4:
        raise RuntimeError("Yeterli hasta verisi toplanamadı.")

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    print(f"\n✅ Toplam {len(X_all)} örnek")

    # ----- VQE -----
    for i, (W_h, W_p) in enumerate(zip(healthy_mats, patho_mats)):
        H_h = build_ising_hamiltonian(W_h)
        H_p = build_ising_hamiltonian(W_p)
        e_h, _, _ = run_vqe(H_h, 5)
        e_p, _, _ = run_vqe(H_p, 5)
        print(f"   {all_subjects[i]}: ΔE={abs(e_h-e_p):.3f}")

    # ----- veri çoğaltma -----
    if len(X_all) < 40:
        rng = np.random.default_rng(42)
        X_aug, y_aug = [], []
        for i in range(len(X_all)):
            X_aug.append(X_all[i])
            y_aug.append(y_all[i])
            for _ in range(4):
                noise = rng.normal(0, 0.01, X_all.shape[1])
                X_aug.append(X_all[i] + noise)
                y_aug.append(y_all[i])
        X_final = np.array(X_aug)
        y_final = np.array(y_aug)
    else:
        X_final = X_all
        y_final = y_all

    # ----- 5‑katlı CV -----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    sonuc = {'SVM':[], 'RF':[], 'VQC':[]}

    for train_idx, test_idx in cv.split(X_final, y_final):
        X_tr, X_te = X_final[train_idx], X_final[test_idx]
        y_tr, y_te = y_final[train_idx], y_final[test_idx]
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        svm = SVC(kernel='rbf').fit(X_tr_s, y_tr)
        pred = svm.predict(X_te_s)
        sonuc['SVM'].append(accuracy_score(y_te, pred))

        rf = RandomForestClassifier(n_estimators=100).fit(X_tr_s, y_tr)
        pred = rf.predict(X_te_s)
        sonuc['RF'].append(accuracy_score(y_te, pred))

        vqc = VQC(
            feature_map=PauliFeatureMap(feature_dimension=6, reps=2, paulis=['Z','ZZ']),
            ansatz=RealAmplitudes(num_qubits=6, reps=2, entanglement='full'),
            sampler=StatevectorSampler(seed=42),
            optimizer=COBYLA(maxiter=200)
        )
        vqc.fit(X_tr[:,:6], y_tr)   # ilk 6 öznitelik
        pred = vqc.predict(X_te[:,:6])
        sonuc['VQC'].append(accuracy_score(y_te, pred))

    print("\n📊 5‑KATLI CV SONUÇLARI")
    for name, accs in sonuc.items():
        print(f"   {name}: {np.mean(accs):.2f} ± {np.std(accs):.2f}")
    print("✅ Tamamlandı.")