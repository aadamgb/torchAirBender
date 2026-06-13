import numpy as np

srt_p   =    [1.288, 1.102, 1.504, 1.015, 1.235]
srt_v   =    [1.592, 1.472, 1.752, 1.358, 1.629]

ctbr_p  =    [0.506, 0.512, 0.506, 0.507, 0.510]
ctbr_v  =    [0.506, 0.506, 0.502, 0.503, 0.504]

lvhr_p  =    [1.17, 1.858, 1.748, 1.32, 1.653]
lvhr_v  =    [1.6, 1.711, 1.524, 1.55, 1.453]

lvhrg_p =    [0.989, 1.044, 0.969, 0.956, 1.016]
lvhrg_v =    [0.716, 0.759, 0.707, 0.698, 0.75]

lists = {
	"srt_p": srt_p,
	"srt_v": srt_v,
	"ctbr_p": ctbr_p,
	"ctbr_v": ctbr_v,
	"lvhr_p": lvhr_p,
	"lvhr_v": lvhr_v,
	"lvhrg_p": lvhrg_p,
	"lvhrg_v": lvhrg_v,
}

print(f"{'list':<8} {'mean':>10} {'std':>10}")
for name, values in lists.items():
	mu = np.mean(values)
	std = np.std(values, ddof=1)
	print(f"{name:<8} {mu:>10.3f} {std:>10.3f}")