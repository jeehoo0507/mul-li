"""
학습된 가중치와 데이터를 JSON으로 묶어 src/index.template.html에 넣고 index.html을 만든다.
사용법:  python tools/build_site.py     (먼저 python sim/sim.py 로 sim/out/params.npz 생성)
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import sim  # noqa: E402

Z = np.load(os.path.join(ROOT, "sim", "out", "params.npz"))
P = {m: [[Z[f"{m}_W{i}"], Z[f"{m}_b{i}"]] for i in range(3)] for m in ("std", "hwa")}


def r(a, d=5):
    return np.round(np.asarray(a, dtype=float), d).tolist()


# 예시 숫자: 열악한 칩(σ=30%)에서 틀리는 숫자 + 숫자별로 골고루
cfg = dict(sim.NOMINAL, sigma=0.3)
wrong = set(np.where(sim.make_chip(P["std"], cfg, 2).run(sim.Xte)[0].argmax(1) != sim.yte)[0].tolist())
gallery = []
for d in range(10):
    idx = np.where(sim.yte == d)[0]
    hard = [int(i) for i in idx if i in wrong]
    easy = [int(i) for i in idx if i not in wrong]
    gallery += (hard[:1] + easy[:2]) if hard else easy[:3]
gallery = gallery[:30]

data = {
    "sizes": sim.SIZES,
    "params": {m: [{"W": r(W), "b": r(b)} for W, b in P[m]] for m in P},
    "Xte": (np.round(sim.Xte * 16)).astype(int).ravel().tolist(),
    "yte": sim.yte.astype(int).tolist(),
    "Xcal": (np.round(sim.XCAL * 16)).astype(int).ravel().tolist(),
    "gallery": gallery,
    "python": {k: json.load(open(os.path.join(ROOT, "sim", "out", "results.json"), encoding="utf-8"))[k]
               for k in ("digital_std", "digital_hwa", "nominal_std", "nominal_hwa")},
}
js = json.dumps(data, separators=(",", ":"))
tpl = open(os.path.join(ROOT, "src", "index.template.html"), encoding="utf-8").read()
assert "__DATA__" in tpl
open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(tpl.replace("__DATA__", js))
print("index.html written,", round(len(js) / 1024), "KB data, gallery", len(gallery))
