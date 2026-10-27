"""
아날로그 인메모리 컴퓨팅 MLP 시뮬레이션
- 실제 MLP(64-32-16-10)를 학습하고, 각 층을 크로스바 회로(G+/G- 차동, TIA, ADC)로 매핑
- 비이상성: 소자 레벨 양자화, 소자 편차, IR drop, 열잡음, 읽기 잡음, DAC/ADC 양자화
- 비교: 디지털 vs 아날로그(일반 학습) vs 아날로그(하드웨어 인식 학습)
numpy만 사용 (sklearn의 내장 digits 데이터셋, 8x8 손글씨 숫자)
"""
import os, json, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# ---------------- style ----------------
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BG, FG, MUTED, GRID = "#000000", "#ececec", "#9a9a9a", "#2a2a2a"
CU, BL, GR, RD = "#ff8a4c", "#6d9bff", "#5fd38d", "#ff5d6c"
CMAP_P = LinearSegmentedColormap.from_list("p", ["#0b0b0b", "#5a2a10", CU, "#ffd2b8"])
CMAP_N = LinearSegmentedColormap.from_list("n", ["#0b0b0b", "#16285a", BL, "#d6e2ff"])


def dark(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#555")
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(color=GRID, lw=0.6)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# ---------------- data ----------------
X, y = load_digits(return_X_y=True)
X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
SIZES = [64, 32, 16, 10]
KB = 1.380649e-23


# ---------------- digital MLP ----------------
def init(seed):
    rng = np.random.default_rng(seed)
    return [[rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)] for a, b in zip(SIZES[:-1], SIZES[1:])]


def forward(params, x, noise=0.0, rng=None):
    acts, pres, Ws = [x], [], []
    h = x
    L = len(params)
    for li, (W, b) in enumerate(params):
        if noise > 0:
            W = W * (1 + noise * rng.standard_normal(W.shape))
            b = b * (1 + noise * rng.standard_normal(b.shape))
        z = h @ W + b
        pres.append(z)
        Ws.append(W)
        h = np.maximum(z, 0) if li < L - 1 else z
        acts.append(h)
    return acts, pres, Ws


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train(seed=0, noise=0.0, epochs=300, lr=3e-3, bs=64, wd=1e-4):
    """noise > 0 이면 하드웨어 인식 학습: 매 배치마다 가중치에 곱셈형 잡음을 넣고 학습"""
    params = init(seed)
    rng = np.random.default_rng(seed + 100)
    m = [[np.zeros_like(p) for p in lay] for lay in params]
    v = [[np.zeros_like(p) for p in lay] for lay in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    n, L = len(Xtr), len(params)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            bi = idx[s:s + bs]
            acts, pres, Ws = forward(params, Xtr[bi], noise, rng)
            g = softmax(acts[-1])
            g[np.arange(len(bi)), ytr[bi]] -= 1
            g /= len(bi)
            grads = [None] * L
            for li in reversed(range(L)):
                grads[li] = [acts[li].T @ g + wd * params[li][0], g.sum(0)]
                if li > 0:
                    g = (g @ Ws[li].T) * (pres[li - 1] > 0)
            t += 1
            for li in range(L):
                for k in range(2):
                    m[li][k] = b1 * m[li][k] + (1 - b1) * grads[li][k]
                    v[li][k] = b2 * v[li][k] + (1 - b2) * grads[li][k] ** 2
                    mh = m[li][k] / (1 - b1 ** t)
                    vh = v[li][k] / (1 - b2 ** t)
                    params[li][k] -= lr * mh / (np.sqrt(vh) + eps)
    return params


def acc_digital(params, x=Xte, yy=yte):
    return float((forward(params, x)[0][-1].argmax(1) == yy).mean())


# ---------------- analog crossbar chip ----------------
NOMINAL = dict(
    Gmax=100e-6, Gmin=1e-6,   # 컨덕턴스 범위 [S]
    levels=32,                # 소자가 표현할 수 있는 컨덕턴스 단계 수 (None = 연속)
    sigma=0.05,               # 소자 편차 (곱셈형, 표준편차)
    beta=0.05,                # IR drop 세기 (가장 먼 교차점의 전압 손실 비율)
    Vread=0.2,                # 입력 최대 전압 [V]
    dac=8, adc=8,             # DAC / ADC 비트 수 (None = 무한 정밀도)
    read=0.002,               # 읽기 잡음 (ADC 풀스케일 대비 표준편차)
    T=300.0, B=1e8,           # 열잡음: 온도 [K], 대역폭 [Hz]
)
IDEAL = dict(NOMINAL, levels=None, sigma=0.0, beta=0.0, dac=None, adc=None, read=0.0, T=0.0)


class Layer:
    pass


class Chip:
    def __init__(self, params, cfg, seed=0, calib=Xtr):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        acts, _, _ = forward(params, calib)
        self.layers = []
        Gmax, Gmin, Lv = cfg["Gmax"], cfg["Gmin"], cfg["levels"]

        def lvl(a):
            return a if Lv is None else np.round(a * (Lv - 1)) / (Lv - 1)

        for li, (W, b) in enumerate(params):
            Ly = Layer()
            Ly.xmax = 1.2 * max(float(acts[li].max()), 1e-6)   # 입력 범위 여유 20%
            Wa = np.vstack([W, b[None, :] / Ly.xmax])          # 바이어스 = 입력이 xmax로 고정된 추가 행
            Ly.wmax = float(np.abs(Wa).max())
            Ly.Gp_t = Gmin + (Gmax - Gmin) * lvl(np.clip(Wa, 0, None) / Ly.wmax)   # 목표 컨덕턴스 (G+)
            Ly.Gn_t = Gmin + (Gmax - Gmin) * lvl(np.clip(-Wa, 0, None) / Ly.wmax)  # 목표 컨덕턴스 (G-)
            s = cfg["sigma"]
            Ly.Gp = np.clip(Ly.Gp_t * (1 + s * self.rng.standard_normal(Wa.shape)), 0, None)  # 실제로 써진 값
            Ly.Gn = np.clip(Ly.Gn_t * (1 + s * self.rng.standard_normal(Wa.shape)), 0, None)
            Nin, Nout = Wa.shape
            i = np.arange(Nin)[:, None]
            j = np.arange(Nout)[None, :]
            # IR drop: 행은 왼쪽에서 구동(오른쪽 열일수록 멀다), 열은 아래에서 읽음(위쪽 행일수록 멀다)
            Ly.a = 1 - cfg["beta"] * 0.5 * ((j + 1) / Nout + (Nin - i) / Nin)
            Ly.k = (Gmax - Gmin) / Ly.wmax
            Ly.s_in = cfg["Vread"] / Ly.xmax
            xin = np.hstack([acts[li], np.full((len(acts[li]), 1), Ly.xmax)])
            Iid = (xin * Ly.s_in) @ (Ly.Gp_t - Ly.Gn_t)
            Ly.FS = 1.2 * float(np.abs(Iid).max())           # ADC 풀스케일 (캘리브레이션, 여유 20%)
            Ly.last = li == len(params) - 1
            self.layers.append(Ly)

    def calibrate(self, params, xcal):
        """열별 보정: 보정용 입력을 흘려서 각 열의 이득(gain)과 오프셋을 측정하고 디지털로 되돌린다.
        층마다 칩이 실제로 받은 입력 h로 '정답' h@W+b를 계산해 최소제곱으로 맞춘다."""
        h = xcal
        for li, Ly in enumerate(self.layers):
            Ly.cal = None
            z = self._layer(Ly, h)
            W, b = params[li]
            tgt = h @ W + b
            g = np.ones(z.shape[1]); o = np.zeros(z.shape[1])
            for j in range(z.shape[1]):
                A = np.vstack([z[:, j], np.ones(len(z))]).T
                g[j], o[j] = np.linalg.lstsq(A, tgt[:, j], rcond=None)[0]
            Ly.cal = (g, o)
            z = z * g + o
            h = z if Ly.last else np.maximum(z, 0)
        return self

    def _layer(self, Ly, h):
        c = self.cfg
        xin = np.hstack([h, np.full((len(h), 1), Ly.xmax)])
        V = np.clip(xin * Ly.s_in, 0, c["Vread"])                          # DAC: 숫자 -> 전압
        if c["dac"]:
            q = 2 ** c["dac"] - 1
            V = np.round(V / c["Vread"] * q) / q * c["Vread"]
        Ip = V @ (Ly.Gp * Ly.a)                                             # 옴 + 키르히호프 (열⁺)
        In = V @ (Ly.Gn * Ly.a)                                             # 옴 + 키르히호프 (열⁻)
        if c["T"] > 0:                                                      # 열잡음 sqrt(4kTB·ΣG)
            Ip = Ip + np.sqrt(4 * KB * c["T"] * c["B"] * Ly.Gp.sum(0)) * self.rng.standard_normal(Ip.shape)
            In = In + np.sqrt(4 * KB * c["T"] * c["B"] * Ly.Gn.sum(0)) * self.rng.standard_normal(In.shape)
        I = Ip - In                                                         # 차동 (뺄셈)
        if c["read"] > 0:
            I = I + c["read"] * Ly.FS * self.rng.standard_normal(I.shape)
        if c["adc"]:                                                        # TIA -> ADC 양자화
            q = Ly.FS / (2 ** (c["adc"] - 1))
            I = np.clip(np.round(I / q) * q, -Ly.FS, Ly.FS)
        z = I / (Ly.k * Ly.s_in)                                            # 숫자로 환산
        return z

    def run(self, x):
        h, pres = x, []
        for Ly in self.layers:
            z = self._layer(Ly, h)
            if getattr(Ly, "cal", None) is not None:
                z = z * Ly.cal[0] + Ly.cal[1]
            pres.append(z)
            h = z if Ly.last else np.maximum(z, 0)
        return h, pres


XCAL = Xtr[:200]


def make_chip(params, cfg, s, cal=False):
    ch = Chip(params, cfg, s)
    return ch.calibrate(params, XCAL) if cal else ch


def acc_chip(params, cfg, seeds, x=Xte, yy=yte, cal=False):
    r = [float((make_chip(params, cfg, s, cal).run(x)[0].argmax(1) == yy).mean()) for s in seeds]
    return float(np.mean(r)), float(np.std(r))


def rel_err(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


# ======================================================================
if __name__ == "__main__":
    print("training...")
    P_std = train(seed=0, noise=0.0)
    P_hwa = train(seed=0, noise=0.15)
    d_std, d_hwa = acc_digital(P_std), acc_digital(P_hwa)
    print("digital acc  std %.4f  hwa %.4f" % (d_std, d_hwa))

    # sanity check: 비이상성을 전부 끄면 아날로그 = 디지털 이어야 함
    ideal_out = Chip(P_std, IDEAL).run(Xte)[0]
    dig_out = forward(P_std, Xte)[0][-1]
    sanity = float(np.max(np.abs(ideal_out - dig_out)))
    print("sanity max|ideal analog - digital| =", sanity)

    SEEDS = list(range(10))
    R = dict(digital_std=d_std, digital_hwa=d_hwa, sanity_maxabs=sanity, nominal=NOMINAL)
    R["nominal_std"] = acc_chip(P_std, NOMINAL, SEEDS)
    R["nominal_hwa"] = acc_chip(P_hwa, NOMINAL, SEEDS)
    R["nominal_std_cal"] = acc_chip(P_std, NOMINAL, SEEDS, cal=True)
    R["nominal_hwa_cal"] = acc_chip(P_hwa, NOMINAL, SEEDS, cal=True)
    print("nominal analog std", R["nominal_std"], "hwa", R["nominal_hwa"])

    def sweep(key, vals, base=NOMINAL):
        out = {"vals": vals, "std": [], "hwa": [], "std_cal": [], "hwa_cal": []}
        for v in vals:
            cfg = dict(base, **{key: v})
            out["std"].append(acc_chip(P_std, cfg, SEEDS))
            out["hwa"].append(acc_chip(P_hwa, cfg, SEEDS))
            out["std_cal"].append(acc_chip(P_std, cfg, SEEDS, cal=True))
            out["hwa_cal"].append(acc_chip(P_hwa, cfg, SEEDS, cal=True))
        print("sweep", key, "done")
        return out

    R["sw_sigma"] = sweep("sigma", [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    R["sw_adc"] = sweep("adc", [2, 3, 4, 5, 6, 7, 8, 10])
    R["sw_beta"] = sweep("beta", [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    R["sw_levels"] = sweep("levels", [2, 4, 8, 16, 32, 64])

    # ---- 오차 원인별 분해 (하나씩만 켜기, 강한 설정) ----
    STRONG = dict(levels=8, sigma=0.15, beta=0.15, adc=4, dac=4, read=0.02, T=300.0)
    names = dict(levels="소자 레벨\n(8단계)", sigma="소자 편차\n(σ=15%)", beta="IR drop\n(β=15%)",
                 adc="ADC\n(4비트)", dac="DAC\n(4비트)", read="읽기 잡음\n(2%)", T="열잡음\n(300K)")
    brk = []
    for key, val in STRONG.items():
        cfg = dict(IDEAL, **{key: val})
        if key == "T":
            cfg["B"] = 1e8
        errs, accs, errc, accc = [], [], [], []
        for s in SEEDS:
            o = Chip(P_std, cfg, s).run(Xte)[0]
            errs.append(rel_err(o, dig_out)); accs.append(float((o.argmax(1) == yte).mean()))
            oc = make_chip(P_std, cfg, s, cal=True).run(Xte)[0]
            errc.append(rel_err(oc, dig_out)); accc.append(float((oc.argmax(1) == yte).mean()))
        brk.append(dict(key=key, name=names[key], err=float(np.mean(errs)), acc=float(np.mean(accs)),
                        err_cal=float(np.mean(errc)), acc_cal=float(np.mean(accc))))
    cfg_all = dict(IDEAL, **STRONG)
    errs, accs, errc, accc = [], [], [], []
    for s in SEEDS:
        o = Chip(P_std, cfg_all, s).run(Xte)[0]
        errs.append(rel_err(o, dig_out)); accs.append(float((o.argmax(1) == yte).mean()))
        oc = make_chip(P_std, cfg_all, s, cal=True).run(Xte)[0]
        errc.append(rel_err(oc, dig_out)); accc.append(float((oc.argmax(1) == yte).mean()))
    brk.append(dict(key="all", name="전부\n동시", err=float(np.mean(errs)), acc=float(np.mean(accs)),
                    err_cal=float(np.mean(errc)), acc_cal=float(np.mean(accc))))
    R["breakdown"] = brk

    # ---- 층별 오차 누적 (nominal) ----
    _, pres_d, _ = forward(P_std, Xte)
    lay_err = {"std": [], "hwa": []}
    _, pres_dh, _ = forward(P_hwa, Xte)
    for s in SEEDS:
        _, pa = Chip(P_std, NOMINAL, s).run(Xte)
        lay_err["std"].append([rel_err(a, b) for a, b in zip(pa, pres_d)])
        _, ph = Chip(P_hwa, NOMINAL, s).run(Xte)
        lay_err["hwa"].append([rel_err(a, b) for a, b in zip(ph, pres_dh)])
    R["layer_err"] = {k: np.mean(v, 0).tolist() for k, v in lay_err.items()}

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(OUT, "params.npz"),
             **{f"std_W{i}": p[0] for i, p in enumerate(P_std)}, **{f"std_b{i}": p[1] for i, p in enumerate(P_std)},
             **{f"hwa_W{i}": p[0] for i, p in enumerate(P_hwa)}, **{f"hwa_b{i}": p[1] for i, p in enumerate(P_hwa)})
    print("saved results")
