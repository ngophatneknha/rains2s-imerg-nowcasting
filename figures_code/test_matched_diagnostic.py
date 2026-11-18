# Diagnostic: recalibrate RainS2S (seed 0) on the TEST set so that bias = 1.000, then compare CSI with persistence.
# Uses test labels, so it is a diagnostic only, not a deployable forecast. Needs p0_lib and the seed-0 test forecasts (4173, 3, 150, 80).
import numpy as np, p0_lib as L
st = L.load_seg('test'); nt = len(st) - 8
P = np.load('seq2seq_unet_s0_test.npy', mmap_mode='r')
for h in (1, 2, 3):
    O = L.obs_mm(st, 0, nt, h); X = L.den(np.asarray(P[:, h - 1])); Pe = L.den(st[5:5 + nt]); K = int((O > 0.5).sum())
    def csi(F, t):
        p, o = F > t, O > 0.5; H = (p & o).sum(); M = (~p & o).sum(); Fa = (p & ~o).sum(); return H / (H + M + Fa), (H + Fa) / (H + M)
    tau = float(np.partition(X.ravel(), -K)[-K]); c, b = csi(X, tau); cp, bp = csi(Pe, 0.5)
    print(h * 30, f'tau={tau:.2f} CSI={c:.4f} bias={b:.3f} | persistence CSI={cp:.4f} bias={bp:.3f} | diff={c - cp:+.4f}')
