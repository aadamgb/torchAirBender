import numpy as np

srt_p   =    [0.233, 0.264, 0.238, 0.187, 0.201]
srt_v   =    [0.189, 0.190, 0.193, 0.190, 0.187]

ctbr_p  =    [0.346, 0.335, 0.329, 0.324, 0.337]
ctbr_v  =    [0.178, 0.179, 0.180, 0.182, 0.179]

lvhr_p  =    [0.137, 0.132, 0.131, 0.131, 0.129]
lvhr_v  =    [0.256, 0.202, 0.214, 0.233, 0.188]

lvhrg_p =    [0.201, 0.202, 0.200, 0.198, 0.206]
lvhrg_v =    [0.219, 0.226, 0.217, 0.221, 0.237]

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