# Hình theo chuẩn xuất bản: vector PDF + PNG 600 dpi, rộng 180 mm (hai cột), Arial 7-8 pt,
# bảng màu Okabe-Ito (an toàn cho người mù màu) + ký hiệu/kiểu nét riêng, không đặt tiêu đề trong hình, có nhãn (a)(b)(c),
# thanh sai số = độ lệch chuẩn qua 3 seed. Mọi số lấy từ output của notebook (c12.txt) và rain_raw.npy.
import re, datetime as dt, numpy as np, matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker, matplotlib.ticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
OUT = '../colab_results/'; W = 7.09   # inch = 180 mm
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.1, 'lines.markersize': 4.5, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial'})
def save(fig, name):
    fig.savefig(OUT + name + '.pdf'); fig.savefig(OUT + name + '.png'); plt.close(fig)
def panel(ax, s, x=-0.14, y=1.04): ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')
UNIT = 'mm h$^{-1}$'

# Okabe-Ito
STY = {  # tên hiển thị, màu, ký hiệu, kiểu nét
    'persistence':    ('Persistence',     '#000000', 'o', '-'),
    'optflow':        ('Dòng quang học',  '#999999', 's', '--'),
    'cnn_gru':        ('CNN–GRU',         '#E69F00', 'v', ':'),
    'convlstm_noatt': ('ConvLSTM',        '#56B4E9', 'D', '-.'),
    'convlstm_att':   ('ConvLSTM+Att',    '#F0E442', 'p', ':'),
    'unet_axial':     ('ConvLSTM–U-Net',  '#009E73', '^', '-'),
    'seq2seq_unet':   ('RainS2S',         '#0072B2', 'P', '-'),
    'simvp_gsta':     ('SimVPv2',         '#D55E00', 'X', '--'),
    'predrnnpp':      ('PredRNN++',       '#CC79A7', '*', '-.'),
}

# ---------- số liệu ----------
t = open('c12.txt', encoding='utf8').read(); a, b = t.split('=== T2.')
def parse(sec):
    m_, s_ = {}, {}
    for l in sec.split('\n'):
        m = re.match(r'\s*(\w+)\s+\+(30|60|90)\s+(.*)', l)
        if not m: continue
        vals = re.findall(r'(-?\d+\.\d+)(?: ± (\d+\.\d+))?', m.group(3)); k = (m.group(1), int(m.group(2)))
        m_[k] = [float(v[0]) for v in vals]; s_[k] = [float(v[1]) if v[1] else 0.0 for v in vals]
    return m_, s_
(T1, S1), (T2, S2) = parse(a), parse(b)   # T1: RMSE MAE CSI POD FAR Bias ... ; T2: CSI POD FAR Bias ...
LEADS = (30, 60, 90)

# ---------- Hình 1: dữ liệu ----------
raw = np.load('../rainfall_processed/rain_raw.npy', mmap_mode='r'); T = len(raw); ntr, nva = int(T * .7), int(T * .15)
mean_map = np.zeros(raw.shape[1:], np.float64); edges = np.linspace(0, 60, 61); hist = np.zeros(60)
for i in range(0, T, 2000):
    x = np.asarray(raw[i:i + 2000]); mean_map += x.sum(0); hist += np.histogram(x[x > 0.1], bins=edges)[0]
mean_map /= T
ts = np.array([np.asarray(raw[i:i + 48]).mean() for i in range(0, T - 47, 48)]); t0 = dt.datetime(2022, 1, 1)
days = [t0 + dt.timedelta(days=k) for k in range(len(ts))]
d_tr, d_va, d_end = [t0 + dt.timedelta(minutes=30 * n) for n in (ntr, ntr + nva, T)]
fig = plt.figure(figsize=(W, 2.45)); gs = fig.add_gridspec(1, 3, width_ratios=[0.62, 0.95, 1.7], wspace=0.6, left=0.065, right=0.975, bottom=0.2, top=0.9)
ax = fig.add_subplot(gs[0]); im = ax.imshow(mean_map, origin='lower', extent=[102, 110, 8, 23], cmap='Blues', aspect='auto')
cb = plt.colorbar(im, ax=ax, pad=0.04); cb.set_label(UNIT); cb.outline.set_linewidth(0.5)
ax.set_xlabel('Kinh độ (°E)'); ax.set_ylabel('Vĩ độ (°N)'); ax.set_xticks([102, 106, 110]); panel(ax, '(a)', -0.35)
for sp in ('top', 'right'): ax.spines[sp].set_visible(True)
ax = fig.add_subplot(gs[1]); ax.bar(edges[:-1], hist / hist.sum(), width=1.0, align='edge', color='#0072B2', linewidth=0)
ax.set_yscale('log'); ax.set_xlabel(f'Cường độ mưa ({UNIT})'); ax.set_ylabel('Tần suất tương đối'); panel(ax, '(b)', -0.3)
ax = fig.add_subplot(gs[2]); ax.plot(days, ts, lw=0.6, color='#000000')
for a_, b_, c_, lb in [(t0, d_tr, '#56B4E9', 'Huấn luyện'), (d_tr, d_va, '#E69F00', 'Kiểm định'), (d_va, d_end, '#CC79A7', 'Kiểm tra')]:
    ax.axvspan(a_, b_, color=c_, alpha=.25, lw=0); ax.text(a_ + (b_ - a_) / 2, 1.36, lb, ha='center', va='top', fontsize=7, rotation=0 if lb == 'Huấn luyện' else 90)
for d_ in (d_tr, d_va): ax.axvline(d_, color='k', lw=0.6, ls='--')
ax.set_ylim(0, 1.4); ax.set_xlim(t0, d_end); ax.set_ylabel(f'Mưa trung bình ngày ({UNIT})'); ax.set_xlabel('Thời gian (tháng/năm)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y')); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); panel(ax, '(c)', -0.12)
save(fig, 'fig_data_overview_vi')

# ---------- Hình 2: quy trình ----------
fig, ax = plt.subplots(figsize=(W, 1.95)); ax.axis('off'); ax.set_xlim(0, 13); ax.set_ylim(0, 3.6)
def box(x, y, w, h, text, fc='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.08', fc=fc, ec='black', lw=0.7)); ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7)
def arrow(x1, y1, x2, y2): ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='black', mutation_scale=7))
box(0.05, 1.25, 2.05, 1.1, 'GPM IMERG Final V07\n30 phút, 0,1°\n01/2022 – 08/2023')
box(2.5, 1.25, 2.05, 1.1, 'Cắt vùng 150 × 80\nlog(1 + x), chuẩn hóa\ntheo tập huấn luyện')
box(4.95, 1.25, 2.05, 1.1, 'Chia theo thời gian\n70 / 15 / 15\n6 khung → 3 khung')
box(7.4, 2.3, 2.35, 1.1, 'Đường cơ sở:\npersistence,\ndòng quang học', '#EEEEEE')
box(7.4, 0.2, 2.35, 1.1, 'Huấn luyện 10 cấu hình\n× 3 seed,\ncùng hàm mất mát')
box(10.15, 0.2, 2.8, 1.1, 'Hiệu chuẩn ngưỡng τ\ntrên tập kiểm định\n(bias = 1)')
box(10.15, 2.3, 2.8, 1.1, 'Đánh giá trên tập kiểm tra:\nPOD, FAR, CSI, Bias, ETS, FSS;\nbootstrap theo khối ngày', '#EEEEEE')
arrow(2.13, 1.8, 2.47, 1.8); arrow(4.58, 1.8, 4.92, 1.8); arrow(7.03, 1.95, 7.37, 2.75); arrow(7.03, 1.65, 7.37, 0.85)
arrow(9.78, 0.75, 10.12, 0.75); arrow(9.78, 2.85, 10.12, 2.85); arrow(11.55, 1.33, 11.55, 2.27)
fig.subplots_adjust(0.005, 0.01, 0.995, 0.99); save(fig, 'fig_pipeline_vi')

# ---------- Hình 3: Bias và POD theo thời hạn ----------
LINE = ['persistence', 'optflow', 'cnn_gru', 'convlstm_noatt', 'unet_axial', 'seq2seq_unet', 'simvp_gsta', 'predrnnpp']
fig, axs = plt.subplots(1, 2, figsize=(W, 2.9)); fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.3, wspace=0.26)
for j, (col_, lab_) in enumerate([(5, 'Độ lệch tần suất (Bias)'), (3, 'POD')]):
    ax = axs[j]
    for m in LINE:
        nm, c_, mk, ls = STY[m]
        ax.errorbar(LEADS, [T1[(m, h)][col_] for h in LEADS], yerr=[S1[(m, h)][col_] for h in LEADS], color=c_, marker=mk, ls=ls, capsize=2, elinewidth=0.6, label=nm,
                    markeredgecolor='black', markeredgewidth=0.3)
    ax.set_xticks(LEADS); ax.set_xlabel('Thời hạn dự báo (phút)'); ax.set_ylabel(lab_); ax.grid(axis='y', lw=0.3, color='#BBBBBB'); panel(ax, '(a)' if j == 0 else '(b)', -0.13)
axs[0].set_yscale('log'); axs[0].set_yticks([1, 2, 3, 4, 5, 6]); axs[0].set_yticklabels(['1', '2', '3', '4', '5', '6']); axs[0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter()); axs[0].tick_params(axis='y', which='minor', length=0); axs[0].axhline(1, color='k', lw=0.5)
h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=4, columnspacing=1.6, handlelength=2.6)
save(fig, 'fig_bias_pod_lead')

# ---------- Hình 4: CSI ngưỡng cố định và sau hiệu chuẩn ----------
BARS = ['persistence', 'optflow', 'cnn_gru', 'convlstm_noatt', 'convlstm_att', 'unet_axial', 'seq2seq_unet', 'simvp_gsta', 'predrnnpp']
fig, axs = plt.subplots(1, 3, figsize=(W, 3.0), sharey=True); fig.subplots_adjust(left=0.065, right=0.995, top=0.93, bottom=0.36, wspace=0.08)
for k, h in enumerate(LEADS):
    ax = axs[k]; x = np.arange(len(BARS)); w = 0.4
    ax.bar(x - w / 2, [T1[(m, h)][2] for m in BARS], w, yerr=[S1[(m, h)][2] for m in BARS], color='#BBBBBB', edgecolor='black', linewidth=0.4,
           error_kw=dict(elinewidth=0.6, capsize=1.5), label='Ngưỡng cố định 0,5 mm h$^{-1}$')
    ax.bar(x + w / 2, [T2[(m, h)][0] for m in BARS], w, yerr=[S2[(m, h)][0] for m in BARS], color='#0072B2', edgecolor='black', linewidth=0.4, hatch='///',
           error_kw=dict(elinewidth=0.6, capsize=1.5), label='Sau hiệu chuẩn theo độ lệch tần suất')
    ax.axhline(T1[('persistence', h)][2], color='black', ls='--', lw=0.7, label='Mức persistence')
    ax.set_xticks(x); ax.set_xticklabels([STY[m][0] for m in BARS], rotation=50, ha='right', rotation_mode='anchor'); ax.set_ylim(0, 0.78)
    ax.grid(axis='y', lw=0.3, color='#BBBBBB'); ax.set_axisbelow(True); ax.text(0.03, 0.96, f'({"abc"[k]}) +{h} phút', transform=ax.transAxes, va='top', fontsize=8, fontweight='bold')
axs[0].set_ylabel('CSI'); h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=3, columnspacing=1.8)
save(fig, 'fig_csi_raw_vs_matched')

# ---------- Hình 5: biểu đồ hiệu năng ----------
fig, axs = plt.subplots(1, 3, figsize=(W, 3.05), sharey=True); fig.subplots_adjust(left=0.065, right=0.995, top=0.93, bottom=0.31, wspace=0.07)
g = np.linspace(0.005, 1, 400); SR, PD = np.meshgrid(g, g); CSI = 1 / (1 / SR + 1 / PD - 1)
for k, h in enumerate(LEADS):
    ax = axs[k]; cs = ax.contour(SR, PD, CSI, levels=np.arange(0.1, 1.0, 0.1), colors='#AAAAAA', linewidths=0.5)
    ax.clabel(cs, levels=[0.2, 0.4, 0.6, 0.8], fmt='%.1f', fontsize=6, inline=True)
    for bb in (0.5, 1, 2, 4):
        ax.plot(g, np.clip(bb * g, 0, 1), ls=':', color='black', lw=0.5); xx = min(1.0, 1 / bb) * 0.9
        ax.text(xx, min(1.0, bb * xx) - 0.045 if bb >= 1 else bb * xx + 0.02, f'{bb:g}', fontsize=6, ha='center', bbox=dict(fc='white', ec='none', pad=0.6))
    for m in LINE:
        nm, c_, mk, ls = STY[m]; r = T1[(m, h)]
        ax.plot(1 - r[4], r[3], marker=mk, ls='none', mfc='white', mec=c_, mew=1.0, ms=5.5, zorder=3)
        if m in ('persistence', 'optflow'): ax.plot(1 - r[4], r[3], marker=mk, ls='none', color=c_, mec='black', mew=0.3, ms=5.5, zorder=3)
        else:
            q = T2[(m, h)]; ax.annotate('', xy=(1 - q[2], q[1]), xytext=(1 - r[4], r[3]), arrowprops=dict(arrowstyle='-', color=c_, lw=0.5, alpha=0.8))
            ax.plot(1 - q[2], q[1], marker=mk, ls='none', color=c_, mec='black', mew=0.3, ms=5.5, zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.set_xlabel('Tỷ lệ thành công (1 − FAR)')
    ax.text(0.03, 0.04, f'({"abc"[k]}) +{h} phút', transform=ax.transAxes, fontsize=8, fontweight='bold', bbox=dict(fc='white', ec='none', pad=1))
    for sp in ('top', 'right'): ax.spines[sp].set_visible(True)
axs[0].set_ylabel('POD')
hd = [Line2D([], [], marker=STY[m][2], ls='none', color=STY[m][1], mec='black', mew=0.3, ms=5.5, label=STY[m][0]) for m in LINE]
hd += [Line2D([], [], marker='o', ls='none', mfc='white', mec='black', ms=5.5, label='Rỗng: ngưỡng cố định'), Line2D([], [], marker='o', ls='none', color='black', ms=5.5, label='Đặc: sau hiệu chuẩn')]
fig.legend(handles=hd, loc='lower center', ncol=5, columnspacing=1.4, handletextpad=0.3)
save(fig, 'fig_performance_diagram')
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
for f in ('fig_data_overview_vi', 'fig_pipeline_vi', 'fig_bias_pod_lead', 'fig_csi_raw_vs_matched', 'fig_performance_diagram'): print(f, Image.open(OUT + f + '.png').size)
