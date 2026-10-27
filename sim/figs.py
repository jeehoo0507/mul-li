"""시각화: 전체 회로도, 컨덕턴스 맵, 정확도 곡선, 오차 분석"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, PathPatch
from matplotlib.path import Path
import sim
from sim import (BG, FG, MUTED, CU, BL, GR, RD, CMAP_P, CMAP_N, dark, savefig, OUT,
                 NOMINAL, IDEAL, Chip, make_chip, forward, Xte, yte, rel_err)

R = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))
Z = np.load(os.path.join(OUT, "params.npz"))
P_std = [[Z[f"std_W{i}"], Z[f"std_b{i}"]] for i in range(3)]
P_hwa = [[Z[f"hwa_W{i}"], Z[f"hwa_b{i}"]] for i in range(3)]

STYLE = {
    "std": dict(c=MUTED, ls="-", m="o", lab="아날로그 (일반 학습)"),
    "hwa": dict(c=CU, ls="-", m="s", lab="아날로그 + 하드웨어 인식 학습"),
    "std_cal": dict(c=BL, ls="-", m="^", lab="아날로그 + 열별 보정"),
    "hwa_cal": dict(c=GR, ls="--", m="D", lab="아날로그 + 둘 다"),
}


# ======================================================================
# 1. 전체 회로도 (MLP 64-32-16-10 전체를 크로스바로)
# ======================================================================
def tri(ax, x, y, s, d, fc, ec, lw=0.6, z=4):
    if d == "r":
        pts = [[x - s, y - s], [x - s, y + s], [x + s, y]]
    else:
        pts = [[x - s, y + s], [x + s, y + s], [x, y - s]]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z))


def bez(ax, p0, p1, p2, p3, color, lw, alpha, z=1):
    path = Path([p0, p1, p2, p3], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw, alpha=alpha, zorder=z))


def shade(base, t):
    t = float(np.clip(t, 0, 1))
    b = np.array(matplotlib_colors_to_rgb(base))
    return tuple(0.12 + (b - 0.12) * t)


def matplotlib_colors_to_rgb(c):
    import matplotlib.colors as mc
    return mc.to_rgb(c)


def sig_mag(z, last):
    if last:
        return (z - z.min()) / (z.max() - z.min() + 1e-9)
    zp = np.maximum(z, 0)
    return zp / (zp.max() + 1e-9)


def draw_full_circuit():
    chip = Chip(P_std, NOMINAL, seed=3)
    sample = 7
    x0 = Xte[sample:sample + 1]
    # 신호 크기 (각 층 입력/출력) - 색 진하기에 사용
    h = x0
    ins, outs = [], []
    for Ly in chip.layers:
        xin = np.hstack([h, np.full((1, 1), Ly.xmax)])[0] / Ly.xmax
        z = chip._layer(Ly, h)
        ins.append(xin)
        outs.append(z[0])
        h = z if Ly.last else np.maximum(z, 0)
    pred = int(np.argmax(outs[-1]))

    fig, ax = plt.subplots(figsize=(26, 11.5), facecolor=BG)
    ax.set_facecolor(BG)
    gap = 2
    blocks = [(0, 0), (82, -16), (132, -24)]   # (x0, ytop)
    out_nodes_prev = None

    # 입력 숫자 이미지 (8x8)
    img = x0[0].reshape(8, 8)
    ix0, iy0, ps = -20.0, -20.0, 1.25
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(ix0, ix0 + 8 * ps, iy0 - 8 * ps, iy0),
              interpolation="nearest", zorder=3)
    ax.add_patch(Rectangle((ix0, iy0 - 8 * ps), 8 * ps, 8 * ps, fc="none", ec="#666", lw=0.8, zorder=4))
    pix = [(ix0 + (k % 8 + 0.5) * ps, iy0 - (k // 8 + 0.5) * ps) for k in range(64)]

    for li, (Ly, (bx, by)) in enumerate(zip(chip.layers, blocks)):
        Gp, Gn = Ly.Gp_t / NOMINAL["Gmax"], Ly.Gn_t / NOMINAL["Gmax"]
        Nin, Nout = Gp.shape
        W = 2 * Nout + gap
        xL, xR = bx - 0.5, bx + W - 0.5
        # 행(입력) 배선 + DAC
        for i in range(Nin):
            yy = by - i
            ax.plot([bx - 1.6, xR + 0.3], [yy, yy], color=CU, lw=0.35, alpha=0.75, zorder=1)
            if i < Nin - 1:
                tri(ax, bx - 2.2, yy, 0.38, "r", shade(CU, ins[li][i]), CU, lw=0.5)
            else:   # 바이어스 행: 고정 전압원
                ax.add_patch(Circle((bx - 2.2, yy), 0.4, fc=shade(GR, 1), ec=GR, lw=0.6, zorder=4))
        # 열(출력) 배선
        cols = [bx + j for j in range(Nout)] + [bx + Nout + gap + j for j in range(Nout)]
        ytia = by - Nin - 1.4
        for cx in cols:
            ax.plot([cx, cx], [by + 0.6, ytia + 0.4], color=BL, lw=0.35, alpha=0.75, zorder=1)
        # 컨덕턴스 셀 (G+, G-)
        ax.imshow(Gp, cmap=CMAP_P, vmin=0, vmax=1, extent=(bx - 0.5, bx + Nout - 0.5, by - Nin + 0.5, by + 0.5),
                  interpolation="nearest", zorder=2, alpha=0.95)
        ax.imshow(Gn, cmap=CMAP_N, vmin=0, vmax=1,
                  extent=(bx + Nout + gap - 0.5, bx + 2 * Nout + gap - 0.5, by - Nin + 0.5, by + 0.5),
                  interpolation="nearest", zorder=2, alpha=0.95)
        for xs in (bx - 0.5, bx + Nout + gap - 0.5):
            ax.add_patch(Rectangle((xs, by - Nin + 0.5), Nout, Nin, fc="none", ec="#444", lw=0.6, zorder=3))
        # TIA (열마다)
        for cx in cols:
            tri(ax, cx, ytia, 0.36, "d", "#13203d", BL, lw=0.5)
        # 차동 증폭기 → ADC → ReLU → 출력
        yd, yadc, yrelu, yo = ytia - 3.6, ytia - 5.6, ytia - 7.4, ytia - 9.0
        mag = sig_mag(outs[li], Ly.last)
        nodes = []
        for j in range(Nout):
            xd = bx + (W - 1) * (j + 0.5) / Nout
            for cx in (bx + j, bx + Nout + gap + j):
                ax.plot([cx, cx, xd], [ytia - 0.4, ytia - 1.4, yd + 0.5], color=FG, lw=0.3, alpha=0.55, zorder=1)
            tri(ax, xd, yd, 0.5, "d", shade(BL, 0.25 + 0.75 * mag[j]), FG, lw=0.5)
            ax.plot([xd, xd], [yd - 0.5, yo], color=FG, lw=0.35, alpha=0.7, zorder=1)
            ax.add_patch(Rectangle((xd - 0.38, yadc - 0.45), 0.76, 0.9, fc=shade(CU, 0.2 + 0.8 * mag[j]),
                                   ec=CU, lw=0.5, zorder=4))
            if not Ly.last:
                tri(ax, xd, yrelu, 0.3, "d", "#2a1408", CU, lw=0.5)
                ax.plot([xd - 0.32, xd + 0.32], [yrelu - 0.32, yrelu - 0.32], color=CU, lw=0.7, zorder=5)
            ax.add_patch(Circle((xd, yo), 0.22, fc=FG, ec="none", zorder=4))
            nodes.append((xd, yo))
        # 이전 층 출력 → 이 층 입력 배선 (곡선 다발)
        if li == 0:
            for k, (px, py) in enumerate(pix):
                bez(ax, (px, py), (px + 6, py), (bx - 9, by - k), (bx - 2.7, by - k), CU, 0.45,
                    0.25 + 0.6 * float(x0[0][k]))
        else:
            for k, (ox, oy) in enumerate(out_nodes_prev):
                yt = by - k
                bez(ax, (ox, oy), (ox, oy - 4), (bx - 14, yt), (bx - 2.7, yt), BL, 0.45, 0.45)
        out_nodes_prev = nodes

    # 최종 출력 10개
    fx = 166
    for k, (ox, oy) in enumerate(out_nodes_prev):
        ty = -26 - k * 2.0
        bez(ax, (ox, oy), (ox, oy - 3), (fx - 8, ty), (fx - 0.6, ty), GR if k == pred else BL,
            0.9 if k == pred else 0.45, 0.95 if k == pred else 0.45)
        if k == pred:
            ax.add_patch(Circle((fx, ty), 0.95, fc=GR, ec="none", alpha=0.25, zorder=4))
        ax.add_patch(Circle((fx, ty), 0.55, fc=GR if k == pred else "#1b2a4a",
                            ec=GR if k == pred else BL, lw=0.8, zorder=5))

    ax.set_xlim(-23, 170)
    ax.set_ylim(-79, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    savefig(fig, "01_full_circuit.png")

    # 확대 버전: 3번째 층만 (세부 구조가 잘 보이게)
    fig, ax2 = plt.subplots(figsize=(13, 10), facecolor=BG)
    ax2.set_facecolor(BG)
    fig2_draw_zoom(ax2, chip.layers[2], ins[2], outs[2], pred)
    savefig(fig, "02_circuit_zoom_layer3.png")
    return pred, int(yte[sample])


def fig2_draw_zoom(ax, Ly, xin, zout, pred):
    Gp, Gn = Ly.Gp_t / NOMINAL["Gmax"], Ly.Gn_t / NOMINAL["Gmax"]
    Nin, Nout = Gp.shape
    gap, bx, by = 2, 0, 0
    W = 2 * Nout + gap
    for i in range(Nin):
        ax.plot([bx - 1.8, bx + W], [by - i, by - i], color=CU, lw=1.1, alpha=0.85, zorder=1)
        if i < Nin - 1:
            tri(ax, bx - 2.5, by - i, 0.4, "r", shade(CU, xin[i]), CU, lw=1)
        else:
            ax.add_patch(Circle((bx - 2.5, by - i), 0.42, fc=GR, ec=GR, zorder=4))
    cols = [bx + j for j in range(Nout)] + [bx + Nout + gap + j for j in range(Nout)]
    ytia = by - Nin - 1.4
    for cx in cols:
        ax.plot([cx, cx], [by + 0.6, ytia + 0.4], color=BL, lw=1.1, alpha=0.85, zorder=1)
    # 셀을 저항 박스로 (색 = 컨덕턴스)
    for i in range(Nin):
        for j in range(Nout):
            for G, cmap, xo in ((Gp, CMAP_P, 0), (Gn, CMAP_N, Nout + gap)):
                ax.add_patch(Rectangle((bx + xo + j - 0.36, by - i - 0.22), 0.72, 0.44,
                                       fc=cmap(G[i, j]), ec="#777", lw=0.4, zorder=3))
    for cx in cols:
        tri(ax, cx, ytia, 0.38, "d", "#13203d", BL, lw=1)
    yd, yadc, yo = ytia - 3.4, ytia - 5.4, ytia - 7.0
    mag = sig_mag(zout, True)
    for j in range(Nout):
        xd = bx + (W - 1) * (j + 0.5) / Nout
        for cx in (bx + j, bx + Nout + gap + j):
            ax.plot([cx, cx, xd], [ytia - 0.4, ytia - 1.3, yd + 0.55], color=FG, lw=0.8, alpha=0.6, zorder=1)
        tri(ax, xd, yd, 0.55, "d", shade(BL, 0.25 + 0.75 * mag[j]), FG, lw=1)
        ax.plot([xd, xd], [yd - 0.55, yo], color=FG, lw=0.9, zorder=1)
        ax.add_patch(Rectangle((xd - 0.4, yadc - 0.5), 0.8, 1.0, fc=shade(CU, 0.2 + 0.8 * mag[j]), ec=CU, lw=1, zorder=4))
        c = GR if j == pred else FG
        ax.add_patch(Circle((xd, yo), 0.32 if j == pred else 0.22, fc=c, ec="none", zorder=4))
    ax.set_xlim(-3.6, W + 0.6)
    ax.set_ylim(yo - 1, by + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")


# ======================================================================
# 2. 컨덕턴스 맵 (목표 vs 실제로 써진 값 vs 차이)
# ======================================================================
def draw_conductance_maps():
    ch = Chip(P_std, dict(NOMINAL, sigma=0.15), seed=1)
    Ly = ch.layers[0]
    Wd = (Ly.Gp_t - Ly.Gn_t) * 1e6
    Wr = (Ly.Gp * Ly.a - Ly.Gn * Ly.a) * 1e6
    fig, axs = plt.subplots(1, 3, figsize=(15, 9), facecolor=BG)
    v = np.abs(Wd).max()
    for ax, M, t, cm, lim in ((axs[0], Wd, "목표 컨덕턴스  G⁺ − G⁻  [µS]", "RdBu_r", v),
                              (axs[1], Wr, "실제 칩 (편차 15% + IR drop)  [µS]", "RdBu_r", v),
                              (axs[2], Wr - Wd, "차이 (오차)  [µS]", "PuOr_r", v * 0.4)):
        im = ax.imshow(M, cmap=cm, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
        dark(ax)
        ax.grid(False)
        ax.set_title(t, fontsize=12, pad=10)
        ax.set_xlabel("출력 열 (뉴런)")
        ax.set_ylabel("입력 행 (픽셀 + 바이어스)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(colors=FG, labelsize=8)
        cb.outline.set_edgecolor("#555")
    fig.subplots_adjust(wspace=0.55)
    fig.suptitle("1층 크로스바 (65×32) 가중치 매핑", color=FG, fontsize=15, y=0.98)
    savefig(fig, "03_conductance_maps.png")


# ======================================================================
# 3. 정확도 곡선
# ======================================================================
def draw_accuracy_curves():
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
    spec = [("sw_sigma", "소자 편차 σ", lambda v: [x * 100 for x in v], "σ [%]"),
            ("sw_adc", "ADC 비트 수", lambda v: v, "비트"),
            ("sw_levels", "소자 컨덕턴스 단계 수", lambda v: v, "단계 수 (log)"),
            ("sw_beta", "IR drop 세기 β", lambda v: [x * 100 for x in v], "가장 먼 셀의 전압 손실 [%]")]
    dig = R["digital_std"] * 100
    for ax, (k, t, fx, xl) in zip(axs.flat, spec):
        xs = fx(R[k]["vals"])
        ax.axhline(dig, color=FG, ls=":", lw=1.2, label="디지털 (기준)")
        for m, st in STYLE.items():
            mu = np.array([a[0] for a in R[k][m]]) * 100
            sd = np.array([a[1] for a in R[k][m]]) * 100
            ax.plot(xs, mu, color=st["c"], ls=st["ls"], marker=st["m"], ms=5, lw=1.8, label=st["lab"])
            ax.fill_between(xs, mu - sd, mu + sd, color=st["c"], alpha=0.12)
        dark(ax)
        ax.set_title(t, fontsize=13, pad=8)
        ax.set_xlabel(xl)
        ax.set_ylabel("테스트 정확도 [%]")
        if k == "sw_levels":
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(v) for v in xs])
        lo = 60 if k == "sw_sigma" else (40 if k in ("sw_adc", "sw_levels") else 95)
        ax.set_ylim(lo, 100)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, labelcolor=FG, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("비이상성별 정확도 (칩 10개 평균 ± 표준편차)", color=FG, fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    savefig(fig, "04_accuracy_curves.png")


# ======================================================================
# 4. 오차 원인별 분해
# ======================================================================
def draw_breakdown():
    B = R["breakdown"]
    names = [d["name"] for d in B]
    e0 = np.array([d["err"] for d in B]) * 100
    e1 = np.array([d["err_cal"] for d in B]) * 100
    a0 = np.array([d["acc"] for d in B]) * 100
    a1 = np.array([d["acc_cal"] for d in B]) * 100
    x = np.arange(len(B))
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.2), facecolor=BG)
    w = 0.38
    ax = axs[0]
    ax.bar(x - w / 2, e0, w, color=MUTED, label="보정 전")
    ax.bar(x + w / 2, e1, w, color=BL, label="열별 보정 후")
    for xi, a, b in zip(x, e0, e1):
        ax.text(xi - w / 2, a + 0.6, f"{a:.0f}", ha="center", color=FG, fontsize=8)
        ax.text(xi + w / 2, b + 0.6, f"{b:.0f}", ha="center", color=BL, fontsize=8)
    dark(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("출력 상대 오차 (RMS) [%]")
    ax.set_title("오차 원인별 출력 오차 (하나씩만 켰을 때)", fontsize=13, pad=8)
    ax.legend(frameon=False, labelcolor=FG)
    ax = axs[1]
    ax.axhline(R["digital_std"] * 100, color=FG, ls=":", lw=1.2, label="디지털")
    ax.bar(x - w / 2, a0, w, color=MUTED, label="보정 전")
    ax.bar(x + w / 2, a1, w, color=BL, label="열별 보정 후")
    dark(ax)
    ax.set_ylim(80, 102)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("테스트 정확도 [%]")
    ax.set_title("오차 원인별 정확도", fontsize=13, pad=8)
    ax.legend(frameon=False, labelcolor=FG, loc="upper center", ncol=3, fontsize=9)
    fig.tight_layout()
    savefig(fig, "05_error_breakdown.png")


# ======================================================================
# 5. IR drop 맵 + 열별 오차
# ======================================================================
def draw_ir_drop():
    cfg = dict(IDEAL, beta=0.15)
    ch = Chip(P_std, cfg, 0)
    Ly = ch.layers[0]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=BG, gridspec_kw=dict(width_ratios=[1, 1.3]))
    im = axs[0].imshow(Ly.a * 100, cmap="inferno", aspect="auto", interpolation="nearest")
    dark(axs[0]); axs[0].grid(False)
    axs[0].set_title("각 교차점에 실제로 걸리는 전압 [% of 입력]", fontsize=12, pad=8)
    axs[0].set_xlabel("열 (오른쪽일수록 구동부에서 멀다)")
    axs[0].set_ylabel("행 (위쪽일수록 읽기부에서 멀다)")
    cb = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors=FG, labelsize=8); cb.outline.set_edgecolor("#555")
    # 열별 출력 오차: 보정 전/후
    _, pd, _ = forward(P_std, Xte)
    _, pa = ch.run(Xte)
    ch.calibrate(P_std, sim.XCAL)
    _, pc = ch.run(Xte)
    e0 = [rel_err(pa[0][:, j], pd[0][:, j]) * 100 for j in range(32)]
    e1 = [rel_err(pc[0][:, j], pd[0][:, j]) * 100 for j in range(32)]
    ax = axs[1]
    ax.plot(range(32), e0, color=MUTED, marker="o", ms=4, lw=1.6, label="보정 전")
    ax.plot(range(32), e1, color=BL, marker="^", ms=4, lw=1.6, label="열별 보정 후")
    dark(ax)
    ax.set_xlabel("출력 열 번호")
    ax.set_ylabel("열 출력 상대 오차 [%]")
    ax.set_title("IR drop(β=15%)만 있을 때 열별 오차", fontsize=12, pad=8)
    ax.legend(frameon=False, labelcolor=FG)
    fig.tight_layout()
    savefig(fig, "06_ir_drop.png")


# ======================================================================
# 6. 층별 오차 누적 + 산점도
# ======================================================================
def draw_layer_and_scatter():
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.4), facecolor=BG, gridspec_kw=dict(width_ratios=[1, 1, 1, 1]))
    ax = axs[0]
    le = R["layer_err"]
    ax.plot([1, 2, 3], np.array(le["std"]) * 100, color=MUTED, marker="o", lw=2, label="일반 학습")
    ax.plot([1, 2, 3], np.array(le["hwa"]) * 100, color=CU, marker="s", lw=2, label="하드웨어 인식 학습")
    dark(ax)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["1층", "2층", "3층(출력)"])
    ax.set_ylabel("층 출력 상대 오차 [%]")
    ax.set_title("층을 지날수록 오차 누적 (기본 설정)", fontsize=12, pad=8)
    ax.legend(frameon=False, labelcolor=FG)
    cfg = dict(NOMINAL, sigma=0.3)
    dig = forward(P_std, Xte)[0][-1]
    digh = forward(P_hwa, Xte)[0][-1]
    cases = [("일반 학습", make_chip(P_std, cfg, 2).run(Xte)[0], dig, MUTED),
             ("하드웨어 인식 학습", make_chip(P_hwa, cfg, 2).run(Xte)[0], digh, CU),
             ("열별 보정", make_chip(P_std, cfg, 2, cal=True).run(Xte)[0], dig, BL)]
    lim = np.abs(dig).max() * 1.05
    for ax, (t, a, d, c) in zip(axs[1:], cases):
        acc = (a.argmax(1) == yte).mean() * 100
        ax.scatter(d.ravel(), a.ravel(), s=3, color=c, alpha=0.35, lw=0)
        ax.plot([-lim, lim], [-lim, lim], color=FG, lw=0.8, ls=":")
        dark(ax)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("디지털 출력 (정답)")
        ax.set_ylabel("아날로그 칩 출력")
        ax.set_title(f"σ=30%  {t}\n오차 {rel_err(a, d) * 100:.0f}%  ·  정확도 {acc:.1f}%", fontsize=11, pad=8)
    fig.tight_layout()
    savefig(fig, "07_layer_error_scatter.png")


# ======================================================================
# 7. 숫자 하나 추론 예시
# ======================================================================
def draw_example():
    cfg = dict(NOMINAL, sigma=0.3)
    a0 = make_chip(P_std, cfg, 2).run(Xte)[0].argmax(1)
    a1 = make_chip(P_std, cfg, 2, cal=True).run(Xte)[0].argmax(1)
    cand = np.where((a0 != yte) & (a1 == yte))[0]
    idx = [int(i) for i in cand[:3]] if len(cand) >= 3 else [7, 21, 44]
    chips = [("디지털", None, FG), ("아날로그 σ=30%", make_chip(P_std, cfg, 2), MUTED),
             ("아날로그 σ=30% + 보정", make_chip(P_std, cfg, 2, cal=True), BL)]
    fig, axs = plt.subplots(len(idx), 2, figsize=(13, 9), facecolor=BG, gridspec_kw=dict(width_ratios=[1, 4]))
    for r, k in enumerate(idx):
        x = Xte[k:k + 1]
        axs[r, 0].imshow(x[0].reshape(8, 8), cmap="gray", interpolation="nearest")
        axs[r, 0].axis("off")
        axs[r, 0].set_title(f"정답 {yte[k]}", color=FG, fontsize=11)
        ax = axs[r, 1]
        w = 0.27
        for c_i, (t, ch, col) in enumerate(chips):
            o = forward(P_std, x)[0][-1][0] if ch is None else ch.run(x)[0][0]
            p = sim.softmax(o[None, :])[0] * 100
            ax.bar(np.arange(10) + (c_i - 1) * w, p, w, color=col, label=t)
        dark(ax)
        ax.set_xticks(range(10))
        ax.set_ylabel("확률 [%]")
        ax.set_ylim(0, 128)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="x", visible=False)
        if r == 0:
            ax.legend(frameon=False, labelcolor=FG, ncol=3, fontsize=9, loc="upper center")
    fig.tight_layout()
    savefig(fig, "08_inference_examples.png")


if __name__ == "__main__":
    pred, true = draw_full_circuit()
    print("full circuit sample pred", pred, "true", true)
    draw_conductance_maps()
    draw_accuracy_curves()
    draw_breakdown()
    draw_ir_drop()
    draw_layer_and_scatter()
    draw_example()
    print(sorted(os.listdir(OUT)))
