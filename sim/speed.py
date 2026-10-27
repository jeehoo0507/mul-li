"""행렬 크기에 따른 한 층 계산 시간: 디지털(메모리 대역폭 한계) vs 아날로그 크로스바 (추정)"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from sim import BG, FG, MUTED, CU, BL, GR, dark, savefig, OUT

BW = 3.35e12            # GPU 메모리 대역폭 [B/s] (최상급 GPU HBM)
BYTES = 1               # 8비트 가중치
PEAK = 2e15             # GPU 8비트 최대 연산 [MAC/s 대략]
TILE = 256              # 크로스바 타일 크기
T_SETTLE, T_ADC, ADC_SHARE, T_SUM = 20e-9, 10e-9, 8, 5e-9

def t_digital(n):       # n×n 행렬 × 벡터, 사용자 1명 (배치 1)
    return max(n * n * BYTES / BW, n * n / PEAK)
def t_analog(n):        # 모든 가중치가 타일에 상주, 타일은 동시에 동작
    tiles_in = int(np.ceil(n / TILE))
    return T_SETTLE + T_ADC * ADC_SHARE + T_SUM * np.log2(max(tiles_in, 1) * 2)

ns = [64, 256, 1024, 4096, 8192, 16384]
rows = [(n, n * n, t_digital(n), t_analog(n), t_digital(n) / t_analog(n), int(np.ceil(n / TILE)) ** 2) for n in ns]
for r in rows: print(r)
json.dump([dict(n=r[0], macs=r[1], t_dig=r[2], t_ana=r[3], speedup=r[4], tiles=r[5]) for r in rows],
          open(os.path.join(OUT, "speed_results.json"), "w"), indent=2)

nn = np.logspace(np.log10(32), np.log10(32768), 200)
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), facecolor=BG)
ax = axs[0]
ax.plot(nn, [t_digital(n) * 1e9 for n in nn], color=MUTED, lw=2.4, label="디지털 (GPU, 가중치를 메모리에서 읽음)")
ax.plot(nn, [t_analog(n) * 1e9 for n in nn], color=CU, lw=2.4, label="아날로그 크로스바 (가중치 제자리)")
ax.set_xscale("log", base=2); ax.set_yscale("log"); dark(ax)
ax.set_xlabel("행렬 크기 n (n×n 한 층)"); ax.set_ylabel("한 층 계산 시간 [ns, log]")
ax.set_title("행렬이 커질수록 디지털은 느려지고 아날로그는 그대로", fontsize=13, pad=10)
ax.axvline(64, color="#444", ls=":", lw=1); ax.text(66, 2e4, "이번 실험\n(64×32)", color=MUTED, fontsize=9.5)
ax.axvline(8192, color="#444", ls=":", lw=1); ax.text(8500*0.38, 2e4, "LLM 한 층\n급", color=MUTED, fontsize=9.5)
ax.legend(frameon=False, labelcolor=FG, fontsize=10, loc="lower right")
ax = axs[1]
lab = [f"{n:,}" for n in ns]; sp = [r[4] for r in rows]
cols = [MUTED if s < 1 else CU for s in sp]
ax.bar(range(len(ns)), sp, color=cols, width=0.6)
ax.set_yscale("log"); ax.axhline(1, color=FG, ls=":", lw=1)
for i, s in enumerate(sp):
    ax.text(i, s * 1.3, f"{s:.2f}배" if s < 10 else f"{s:,.0f}배", ha="center", color=FG, fontsize=11)
ax.set_xticks(range(len(ns))); ax.set_xticklabels(lab); dark(ax); ax.grid(axis="x", visible=False)
ax.set_xlabel("행렬 크기 n"); ax.set_ylabel("아날로그가 빠른 배수 [log]")
ax.set_title("속도 배수 (디지털 시간 ÷ 아날로그 시간)", fontsize=13, pad=10)
ax.set_ylim(0.004, 3e4)
fig.text(0.5, -0.02, "추정치 · 디지털: 대역폭 3.35 TB/s, 8비트 가중치, 배치 1  |  아날로그: 256×256 타일 동시 동작, 정착 20 ns, ADC 10 ns × 8회 공유",
         ha="center", color=MUTED, fontsize=9.5)
fig.tight_layout()
savefig(fig, "13_speed_scaling.png")
