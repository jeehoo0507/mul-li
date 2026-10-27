"""연산량·시간·에너지 비교: 디지털 MLP vs 아날로그 크로스바 칩"""
import time, json, os
import numpy as np
import sim
from sim import forward, Chip, make_chip, NOMINAL, Xte, SIZES

Z = np.load(os.path.join(sim.OUT, "params.npz"))
P = [[Z[f"std_W{i}"], Z[f"std_b{i}"]] for i in range(3)]
layers = list(zip(SIZES[:-1], SIZES[1:]))           # (64,32),(32,16),(16,10)

# ---------- 1. 연산 횟수 ----------
mac = sum(a * b for a, b in layers)                 # 곱셈·덧셈 쌍
bias = sum(b for _, b in layers)
relu = sum(b for _, b in layers[:-1])
dig_flops = 2 * mac + bias + relu
params = sum(a * b + b for a, b in layers)
cells = sum(2 * (a + 1) * b for a, b in layers)      # G+, G- 저항 수
outs = sum(b for _, b in layers)                     # 열(출력) 수
dac = sum(a + 1 for a, _ in layers)                  # 바이어스 행 포함
tia = 2 * outs
adc = outs
cal_ops = 2 * outs                                   # 출력마다 곱셈 1 + 덧셈 1
ana_digital_ops = relu + cal_ops

# ---------- 2. 디지털 실측 (CPU, numpy) ----------
def bench(fn, reps):
    fn()
    t = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t) / reps

x1 = Xte[:1]
t_dig_1 = bench(lambda: forward(P, x1), 20000)
t_dig_b = bench(lambda: forward(P, Xte), 2000) / len(Xte)
ch = Chip(P, NOMINAL, 0)
t_sim_1 = bench(lambda: ch.run(x1), 2000)

# ---------- 3. 아날로그 추정 ----------
# 가정 (공개 문헌의 대표값, 범위가 넓음)
T_SETTLE = 20e-9        # 크로스바 정착 (RC) + TIA 응답, 층당
T_ADC = 10e-9           # 8비트 SAR ADC 한 번 변환
ADC_SHARE = 8           # ADC 하나를 열 8개가 번갈아 씀
T_DIG = 5e-9            # 층당 ReLU·보정·DAC 갱신 (디지털)
ana_lat = []
for (a, b) in layers:
    conv = 2 * b                                   # G+ 열, G- 열 각각? -> 차동 후 1회
    conv = b
    t = T_SETTLE + T_ADC * int(np.ceil(conv / min(ADC_SHARE, conv))) * 1 + T_DIG
    # 공유 시 순차 변환 횟수 = 열 수 / ADC 개수
    n_adc = int(np.ceil(b / ADC_SHARE))
    t = T_SETTLE + T_ADC * int(np.ceil(b / n_adc)) + T_DIG
    ana_lat.append(t)
t_ana = sum(ana_lat)

# 에너지
E_MULT, E_ADD = 3.7e-12, 0.9e-12    # 45nm 32비트 부동소수 곱셈·덧셈 (Horowitz 2014)
E_SRAM = 5e-12                      # 32비트 SRAM 읽기 (작은 캐시, 대략)
E_DRAM = 640e-12                    # 32비트 DRAM 읽기
e_dig_compute = mac * (E_MULT + E_ADD) + (bias + relu) * E_ADD
e_dig_sram = e_dig_compute + params * E_SRAM
e_dig_dram = e_dig_compute + params * E_DRAM

# 아날로그: 실제 칩 시뮬레이션의 전압·컨덕턴스로 저항에서 소모되는 전력 계산
h = x1; e_cell = 0.0
for Ly in ch.layers:
    xin = np.hstack([h, np.full((1, 1), Ly.xmax)])
    V = np.clip(xin * Ly.s_in, 0, NOMINAL["Vread"])[0]
    p = (V ** 2) @ (Ly.Gp + Ly.Gn)                 # 열마다 Σ V²G  [W]
    e_cell += p.sum() * T_SETTLE
    z = ch._layer(Ly, h); h = z if Ly.last else np.maximum(z, 0)
FOM = 20e-15                        # ADC 효율 (J / 변환단계), Walden FOM 대표값
E_ADC = FOM * 2 ** 8
E_DAC = 0.5e-12
E_TIA = 0.2e-12                     # TIA·차동 증폭기 한 번 (대략)
e_ana = e_cell + adc * E_ADC + dac * E_DAC + tia * E_TIA + ana_digital_ops * E_ADD
breakdown = dict(cell=e_cell, adc=adc * E_ADC, dac=dac * E_DAC, tia=tia * E_TIA, digital=ana_digital_ops * E_ADD)

R = dict(mac=mac, dig_flops=dig_flops, params=params, cells=cells, outs=outs, dac=dac, tia=tia, adc=adc,
         ana_digital_ops=ana_digital_ops, t_dig_1=t_dig_1, t_dig_b=t_dig_b, t_sim_1=t_sim_1,
         ana_lat_layers=ana_lat, t_ana=t_ana, e_dig_sram=e_dig_sram, e_dig_dram=e_dig_dram,
         e_dig_compute=e_dig_compute, e_ana=e_ana, breakdown=breakdown)
json.dump(R, open(os.path.join(sim.OUT, "cost_results.json"), "w"), indent=2)
for k, v in R.items(): print(k, v)
