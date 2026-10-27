"""실험 정리용 추가 그림: 데이터·구조, 학습 곡선, 가중치→컨덕턴스 변환, 한 숫자의 신호 흐름, 칩별 편차"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import sim
from sim import (BG, FG, MUTED, CU, BL, GR, RD, dark, savefig, OUT, NOMINAL, Chip, make_chip,
                 forward, train, Xtr, ytr, Xte, yte, SIZES)

Z = np.load(os.path.join(OUT, "params.npz"))
P_std = [[Z[f"std_W{i}"], Z[f"std_b{i}"]] for i in range(3)]
P_hwa = [[Z[f"hwa_W{i}"], Z[f"hwa_b{i}"]] for i in range(3)]


# ---------------- A. 데이터와 MLP 구조 ----------------
def fig_data_mlp():
    fig = plt.figure(figsize=(16, 5.6), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], wspace=0.08)
    sub = gs[:, 0].subgridspec(4, 10, wspace=0.12, hspace=0.05)
    for d in range(10):
        idx = np.where(ytr == d)[0][:4]
        for r, i in enumerate(idx):
            ax = fig.add_subplot(sub[r, d])
            ax.imshow(Xtr[i].reshape(8, 8), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#333")
            if r == 0:
                ax.set_title(str(d), color=FG, fontsize=11)
    ax = fig.add_subplot(gs[:, 1])
    ax.set_facecolor(BG); ax.axis("off")
    xs = [0, 3.2, 6.4, 9.6]
    labels = ["입력\n64 (8×8 픽셀)", "은닉 1\n32", "은닉 2\n16", "출력\n10 (숫자)"]
    show = [16, 12, 9, 10]
    pos = []
    for li, (x, n, k) in enumerate(zip(xs, SIZES, show)):
        ys = np.linspace(-4.2, 4.2, k)
        pos.append([(x, y) for y in ys])
        col = CU if li == 0 else (GR if li == 3 else BL)
        for y in ys:
            ax.add_patch(Circle((x, y), 0.2, fc=BG, ec=col, lw=1.4, zorder=3))
        ax.text(x, -5.4, labels[li], color=FG, ha="center", va="top", fontsize=11)
        if n > k:
            ax.text(x + 0.35, 0, "⋮", color=MUTED, fontsize=14, va="center")
    for a, b in zip(pos[:-1], pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1 + .2, x2 - .2], [y1, y2], color="#3a3a3a", lw=0.35, zorder=1)
    for li in range(3):
        xm = (xs[li] + xs[li + 1]) / 2
        nin, nout = SIZES[li] + 1, SIZES[li + 1]
        ax.text(xm, 4.95, f"크로스바 {nin}×{nout}\n(G⁺, G⁻ 두 배열)", color=MUTED, ha="center", va="bottom", fontsize=9.5)
        ax.text(xm, -4.9, "ReLU" if li < 2 else "argmax", color=MUTED, ha="center", fontsize=9)
    ax.set_xlim(-1, 10.6); ax.set_ylim(-7, 6.4)
    fig.suptitle("데이터: 8×8 손글씨 숫자 1797장 (학습 1257 / 테스트 540)        신경망: 64 → 32 → 16 → 10 MLP",
                 color=FG, fontsize=13, y=0.99)
    savefig(fig, "00a_data_and_mlp.png")


# ---------------- B. 학습 곡선 ----------------
def fig_training():
    logs = {}
    for m, nz in (("std", 0.0), ("hwa", 0.15)):
        lg = {}
        P = train(seed=0, noise=nz, log=lg)
        ref = P_std if m == "std" else P_hwa
        same = all(np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]) for a, b in zip(P, ref))
        print(m, "weights identical to saved:", same)
        logs[m] = lg
    ep = np.arange(1, len(logs["std"]["train_loss"]) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    for m, c, lab in (("std", MUTED, "일반 학습"), ("hwa", CU, "하드웨어 인식 학습 (가중치 잡음 15%)")):
        axs[0].plot(ep, logs[m]["train_loss"], color=c, lw=1.6, label=f"{lab} · 학습")
        axs[0].plot(ep, logs[m]["test_loss"], color=c, lw=1.6, ls="--", label=f"{lab} · 테스트")
        axs[1].plot(ep, np.array(logs[m]["test_acc"]) * 100, color=c, lw=1.8, label=lab)
    dark(axs[0]); dark(axs[1])
    axs[0].set_yscale("log"); axs[0].set_xlabel("에포치"); axs[0].set_ylabel("교차 엔트로피 손실 (log)")
    axs[0].set_title("손실", fontsize=13, pad=8)
    axs[1].set_xlabel("에포치"); axs[1].set_ylabel("테스트 정확도 [%]"); axs[1].set_ylim(80, 100)
    axs[1].set_title("테스트 정확도 (디지털)", fontsize=13, pad=8)
    axs[0].legend(frameon=False, labelcolor=FG, fontsize=9)
    axs[1].legend(frameon=False, labelcolor=FG, fontsize=10, loc="lower right")
    fig.tight_layout()
    savefig(fig, "00b_training_curve.png")
    return {m: logs[m]["test_acc"][-1] for m in logs}


# ---------------- C. 가중치 → 컨덕턴스 ----------------
def fig_weight_to_G():
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=BG, gridspec_kw=dict(width_ratios=[1, 1.1, 1.25]))
    ax = axs[0]
    for li, c in zip(range(3), [CU, BL, GR]):
        ax.hist(P_std[li][0].ravel(), bins=60, color=c, alpha=0.55, label=f"{li + 1}층 ({P_std[li][0].size}개)", density=True)
    dark(ax); ax.set_xlabel("가중치 w"); ax.set_ylabel("밀도"); ax.set_title("학습된 가중치 분포", fontsize=13, pad=8)
    ax.legend(frameon=False, labelcolor=FG)
    ax = axs[1]
    w = np.linspace(-1, 1, 801)
    Gmax, Gmin = NOMINAL["Gmax"] * 1e6, NOMINAL["Gmin"] * 1e6
    for L, ls, a in ((None, "-", 1.0), (8, "-", 0.9)):
        q = (lambda v: v) if L is None else (lambda v, L=L: np.round(v * (L - 1)) / (L - 1))
        gp = Gmin + (Gmax - Gmin) * q(np.clip(w, 0, None))
        gn = Gmin + (Gmax - Gmin) * q(np.clip(-w, 0, None))
        lab = "연속" if L is None else f"{L}단계"
        ax.plot(w, gp, color=CU, ls=ls, lw=2.2 if L is None else 1.4, alpha=a if L is None else .7, label=f"G⁺ ({lab})")
        ax.plot(w, gn, color=BL, ls=ls, lw=2.2 if L is None else 1.4, alpha=a if L is None else .7, label=f"G⁻ ({lab})")
    dark(ax); ax.set_xlabel("가중치 w / w_max"); ax.set_ylabel("컨덕턴스 [µS]")
    ax.set_title("변환 규칙: 양수는 G⁺, 음수는 G⁻\nG = G_min + (G_max − G_min)·|w| / w_max", fontsize=12, pad=8)
    ax.legend(frameon=False, labelcolor=FG, fontsize=9, ncol=2, loc="upper center")
    ax.set_ylim(-3, 118)
    ax = axs[2]
    ch = Chip(P_std, dict(NOMINAL, sigma=0, beta=0), 0)
    Ly = ch.layers[2]
    Wa = np.vstack([P_std[2][0], P_std[2][1][None, :] / Ly.xmax])[:8, :8]
    Gp, Gn = Ly.Gp_t[:8, :8] * 1e6, Ly.Gn_t[:8, :8] * 1e6
    ax.set_facecolor(BG); ax.axis("off")
    for k, (M, cmap, t, x0) in enumerate(((Wa, "RdBu_r", "가중치 w", 0), (Gp, None, "G⁺ [µS]", 9.5), (Gn, None, "G⁻ [µS]", 19))):
        for i in range(8):
            for j in range(8):
                v = M[i, j]
                if cmap:
                    lim = np.abs(Wa).max(); col = plt.get_cmap(cmap)(0.5 + 0.5 * v / lim)
                else:
                    col = (sim.CMAP_P if k == 1 else sim.CMAP_N)(v / Gmax)
                ax.add_patch(FancyBboxPatch((x0 + j, -i), 0.9, 0.9, boxstyle="round,pad=0,rounding_size=0.08", fc=col, ec="#222", lw=.5))
                if k > 0 and v > 15:
                    ax.text(x0 + j + .45, -i + .45, f"{v:.0f}", color="black" if v > 55 else FG, ha="center", va="center", fontsize=6.5)
        ax.text(x0 + 4, 1.4, t, color=FG, ha="center", fontsize=11)
    ax.text(8.75, -3.1, "→", color=FG, fontsize=20, ha="center", va="center")
    ax.text(18.25, -3.1, "+", color=FG, fontsize=18, ha="center", va="center")
    ax.set_xlim(-0.3, 27.2); ax.set_ylim(-7.6, 2.4); ax.set_aspect("equal")
    ax.set_title("3층 일부 (8×8): 가중치 하나가 저항 두 개로", color=FG, fontsize=13, pad=8)
    fig.tight_layout()
    savefig(fig, "09_weight_to_conductance.png")


# ---------------- D. 한 숫자의 신호 흐름 ----------------
def fig_signal_flow(sample=7):
    ch = Chip(P_std, NOMINAL, 3)
    x = Xte[sample:sample + 1]
    _, pres_d, _ = forward(P_std, x)
    fig, axs = plt.subplots(3, 3, figsize=(17, 11), facecolor=BG, gridspec_kw=dict(width_ratios=[1, 1.6, 1.6]))
    h = x
    for li, Ly in enumerate(ch.layers):
        xin = np.hstack([h, np.full((1, 1), Ly.xmax)])
        V = np.clip(xin * Ly.s_in, 0, NOMINAL["Vread"])[0] * 1e3
        Ip = (V / 1e3) @ (Ly.Gp * Ly.a) * 1e6
        In = (V / 1e3) @ (Ly.Gn * Ly.a) * 1e6
        z = ch._layer(Ly, h)[0]
        ax = axs[li, 0]
        cols = [CU] * (len(V) - 1) + [GR]
        ax.barh(np.arange(len(V)), V, color=cols, height=0.8)
        ax.invert_yaxis(); dark(ax)
        ax.set_xlabel("행 전압 [mV]"); ax.set_ylabel("입력 행 (마지막 = 바이어스)")
        ax.set_title(f"{li + 1}층 ① DAC 입력 전압", fontsize=12, pad=6)
        ax = axs[li, 1]
        j = np.arange(len(Ip)); w = 0.4
        ax.bar(j - w / 2, Ip, w, color=CU, label="I⁺ = ΣG⁺V")
        ax.bar(j + w / 2, In, w, color=BL, label="I⁻ = ΣG⁻V")
        dark(ax); ax.set_xlabel("출력 열"); ax.set_ylabel("열 전류 [µA]")
        ax.set_title(f"{li + 1}층 ② 옴 + 키르히호프: 열마다 모인 전류", fontsize=12, pad=6)
        ax.legend(frameon=False, labelcolor=FG, fontsize=9)
        ax = axs[li, 2]
        zd = pres_d[li][0]
        cz = [FG if v >= 0 else MUTED for v in z]
        ax.bar(j, z, 0.6, color=cz, label="칩 출력 (차동 + ADC)")
        ax.scatter(j, zd, marker="_", s=180, color=BL, lw=2.2, label="디지털 정답", zorder=3)
        ax.axhline(0, color="#555", lw=.8)
        dark(ax); ax.set_xlabel("뉴런"); ax.set_ylabel("출력값 z")
        t = "③ I⁺ − I⁻ → ADC → 숫자" + ("  (음수는 ReLU로 0)" if li < 2 else "  (가장 큰 값이 답)")
        ax.set_title(f"{li + 1}층 {t}", fontsize=12, pad=6)
        ax.legend(frameon=False, labelcolor=FG, fontsize=9)
        h = z[None, :] if Ly.last else np.maximum(z, 0)[None, :]
    fig.suptitle(f"테스트 숫자 하나(정답 {yte[sample]})가 칩을 지나는 과정  ·  보통 칩 설정, 예측 {int(np.argmax(h))}",
                 color=FG, fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    savefig(fig, "10_signal_flow_one_digit.png")


# ---------------- E. 칩별 편차 ----------------
def fig_chip_spread(n=40):
    seeds = range(100, 100 + n)
    res = {}
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), facecolor=BG)
    for ax, sg in zip(axs, (0.15, 0.30)):
        cfg = dict(NOMINAL, sigma=sg)
        r = {}
        for m, P, cal, c, lab in (("std", P_std, False, MUTED, "일반 학습"), ("hwa", P_hwa, False, CU, "하드웨어 인식 학습"),
                                  ("cal", P_std, True, BL, "열별 보정")):
            acc = np.array([float((make_chip(P, cfg, s, cal).run(Xte)[0].argmax(1) == yte).mean()) for s in seeds]) * 100
            r[m] = acc
            bins = np.arange(90, 100.25, 0.5) if sg < 0.2 else np.arange(70, 100.5, 1)
            ax.hist(acc, bins=bins, color=c, alpha=0.6, label=f"{lab}  평균 {acc.mean():.1f}%, 최저 {acc.min():.1f}%")
        ax.axvline(sim.acc_digital(P_std) * 100, color=FG, ls=":", lw=1.2)
        dark(ax); ax.set_xlabel("테스트 정확도 [%]"); ax.set_ylabel(f"칩 수 (총 {n}개)")
        ax.set_title(f"소자 편차 σ = {int(sg * 100)}%: 칩 {n}개의 정확도 분포", fontsize=13, pad=8)
        ax.legend(frameon=False, labelcolor=FG, fontsize=9.5, loc="upper left")
        res[sg] = {k: [float(v.mean()), float(v.min()), float(v.max())] for k, v in r.items()}
    fig.tight_layout()
    savefig(fig, "11_chip_to_chip_spread.png")
    return res


if __name__ == "__main__":
    fig_data_mlp()
    accs = fig_training()
    fig_weight_to_G()
    fig_signal_flow()
    spread = fig_chip_spread()
    json.dump({"final_test_acc": accs, "chip_spread": {str(k): v for k, v in spread.items()}},
              open(os.path.join(OUT, "extra_results.json"), "w"), indent=2)
    print(accs, spread)
