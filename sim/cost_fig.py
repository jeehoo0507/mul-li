import json, os
import numpy as np
import matplotlib.pyplot as plt
from sim import BG, FG, MUTED, CU, BL, GR, dark, savefig, OUT
R = json.load(open(os.path.join(OUT, "cost_results.json")))

fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), facecolor=BG, gridspec_kw=dict(width_ratios=[1.35, 1]))

# 1) 숫자 1개당 에너지 (log)
labels = ["디지털 32비트\n가중치 DRAM", "디지털 32비트\n가중치 SRAM", "디지털 32비트\n연산만", "디지털 8비트\n연산만", "아날로그\n크로스바"]
vals = np.array([R["e_dig_dram"], R["e_dig_sram"], R["e_dig_compute"], 2720 * (0.2e-12 + 0.03e-12), R["e_ana"]]) * 1e9
cols = [MUTED, MUTED, MUTED, MUTED, CU]
ax = axs[0]
y = np.arange(len(vals))[::-1]
ax.barh(y, vals, color=cols, height=0.62)
ax.set_xscale("log")
for yi, v in zip(y, vals):
    ax.text(v * 1.25, yi, f"{v:,.2f} nJ" if v < 10 else f"{v:,.0f} nJ", va="center", color=FG, fontsize=11)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)
dark(ax); ax.grid(axis="y", visible=False)
ax.set_xlim(0.1, 2e4)
ax.set_xlabel("숫자 1개 추론 에너지 [nJ, log]")
ax.set_title("에너지: 가중치를 옮기지 않으면 수천 배 차이", fontsize=13, pad=10)

# 2) 아날로그 에너지 내역
b = R["breakdown"]
parts = [("ADC", b["adc"], CU), ("디지털 처리\n(ReLU·보정)", b["digital"], MUTED), ("DAC", b["dac"], "#c9a27a"),
         ("TIA", b["tia"], BL), ("저항 격자", b["cell"], GR)]
tot = sum(p[1] for p in parts)
ax = axs[1]
ax.set_facecolor(BG); ax.axis("off")
left = 0
for name, v, c in parts:
    w = v / tot
    ax.barh(0, w, left=left, color=c, height=0.5, edgecolor=BG, lw=2)
    if w > 0.08:
        ax.text(left + w / 2, 0, f"{w*100:.0f}%", ha="center", va="center", color="black", fontsize=13, fontweight="bold")
    left += w
for i, (name, v, c) in enumerate(parts):
    ax.add_patch(plt.Rectangle((0.0 + (i % 2) * 0.5, -0.62 - (i // 2) * 0.22), 0.03, 0.1, color=c, transform=ax.transData))
    ax.text(0.045 + (i % 2) * 0.5, -0.57 - (i // 2) * 0.22, f"{name.replace(chr(10), ' ')}  {v/tot*100:.0f}%", color=FG, va="center", fontsize=10.5)
ax.annotate("계산하는 저항은 2%", xy=(1 - b["cell"] / tot / 2, 0.25), xytext=(0.72, 0.62), color=GR, fontsize=12,
            arrowprops=dict(arrowstyle="->", color=GR, lw=1.2))
ax.set_xlim(0, 1); ax.set_ylim(-1.2, 0.9)
ax.set_title(f"아날로그 칩 에너지 내역 ({R['e_ana']*1e9:.2f} nJ)", color=FG, fontsize=13, pad=10)

fig.text(0.5, -0.02, "디지털 연산 2,720 MAC · 아날로그 칩은 MAC 0회 (저항이 전류로 계산)   |   아날로그 값은 대표 소자 수치로 계산한 추정치",
         ha="center", color=MUTED, fontsize=10)
fig.tight_layout()
savefig(fig, "12_cost_key.png")
