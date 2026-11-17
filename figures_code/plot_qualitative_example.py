# Hình 6 (định tính) theo chuẩn xuất bản: quan sát và dự báo Seq2Seq (seed 0) cho 2 mẫu của tập kiểm tra.
# Mẫu được chọn theo quy tắc cố định (20% và 55% chiều dài tập kiểm tra), không chọn theo kết quả.
import datetime as dt, numpy as np, matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
OUT = '../colab_results/'; W = 7.09; LOG_MAX = 4.684628009796143
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6, 'pdf.fonttype': 42, 'savefig.dpi': 600, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial'})
P = np.load('preds/seq2seq_unet_s0_test.npy', mmap_mode='r', allow_pickle=False); n = len(P); assert P.shape == (4173, 3, 150, 80)
raw = np.load('../rainfall_processed/rain_raw.npy', mmap_mode='r'); T = len(raw); a0 = int(T * .7) + int(T * .15); t0 = dt.datetime(2022, 1, 1)
den = lambda x: np.expm1(np.asarray(x, np.float32) * LOG_MAX)
levels = [0.1, 0.5, 1, 2, 5, 10, 20, 50]; cmap = plt.get_cmap('viridis', len(levels) - 1).copy(); cmap.set_under('white'); cmap.set_over(plt.get_cmap('viridis')(1.0)); norm = BoundaryNorm(levels, cmap.N)
cases = [int(0.2 * n), int(0.55 * n)]
fig = plt.figure(figsize=(W, 4.0)); gs = fig.add_gridspec(2, 9, width_ratios=[1, 1, 1, 1, 0.42, 1, 1, 1, 1], left=0.075, right=0.995, top=0.9, bottom=0.25, wspace=0.1, hspace=0.1)
for ci, i in enumerate(cases):
    tt = t0 + dt.timedelta(minutes=30 * (a0 + i + 5)); c0 = ci * 5
    for col in range(4):
        ao, ap = fig.add_subplot(gs[0, c0 + col]), fig.add_subplot(gs[1, c0 + col])
        obs = np.asarray(raw[a0 + i + 5 + col]); im = ao.imshow(obs, origin='lower', extent=[102, 110, 8, 23], cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        ao.set_title('t (đầu vào)' if col == 0 else f't + {col * 30} phút', fontsize=7, pad=2)
        for ax in (ao, ap): ax.set_xticks([104, 108]); ax.set_yticks([10, 15, 20]); ax.tick_params(labelbottom=False, labelleft=False)
        if col == 0:
            ap.axis('off'); ap.text(0.5, 0.5, 'Dự báo\nRainS2S', transform=ap.transAxes, ha='center', va='center', fontsize=7, fontweight='bold')
            ao.tick_params(labelbottom=True, labelleft=True)
            if ci == 0: ao.set_ylabel('Vĩ độ (°N)', fontsize=7)
            fig.text(ao.get_position().x0, 0.955, f'({"ab"[ci]}) t = {tt:%d/%m/%Y %H:%M} UTC', fontsize=8, fontweight='bold', va='bottom')
        else:
            ap.imshow(den(P[i, col - 1]), origin='lower', extent=[102, 110, 8, 23], cmap=cmap, norm=norm, aspect='auto', interpolation='nearest'); ap.tick_params(labelbottom=True, labelleft=(col == 1))
fig.text(0.535, 0.165, 'Kinh độ (°E)', ha='center', fontsize=7)
cax = fig.add_axes([0.3, 0.085, 0.47, 0.022]); cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend='max', ticks=levels)
cb.ax.set_xticklabels(['0,1', '0,5', '1', '2', '5', '10', '20', '50']); cb.set_label('Cường độ mưa (mm h$^{-1}$)', fontsize=7, labelpad=1); cb.outline.set_linewidth(0.4)
fig.savefig(OUT + 'fig_qualitative_seq2seq.pdf'); fig.savefig(OUT + 'fig_qualitative_seq2seq.png'); plt.close(fig)
from PIL import Image; Image.MAX_IMAGE_PIXELS = None; print(Image.open(OUT + 'fig_qualitative_seq2seq.png').size, cases)
