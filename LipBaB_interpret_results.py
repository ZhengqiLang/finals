import numpy as np
np.set_printoptions(suppress=True)

def round_lip(lip):
    if lip > 1000: return round(lip, 0)
    else: return round(lip, 2)

def print_line(arr, name, prob, netw):
    print(str(name), str(prob), str(netw), *[round_lip(x) for x in [arr[0], arr[3], arr[6], arr[7]]], 
             *[("\\textgreater " if arr[9]>0 else "")+str(round(arr[5], 1)), 
               ("\\textgreater " if arr[10]>0 else "")+str(int(arr[8]+0.5))], sep=' & ', end = " \\\\\n")
    
results_policy = []
results_certificate = []
names = []
probabilities = []

types = 8
seeds = 10
timeout = 600
for _ in range(types):
    policy = np.zeros(11)
    certificate = np.zeros(11)
    nresults = 0
    for _ in range(seeds):
        name, probability, _ = input().split()
        check_skip = input().strip()
        if check_skip == "skip":
            continue
        else: nresults += 1
        while input()[0] == "-": pass  # for the first network (policy network), we have to skip the first lines of the output (the output of the checkpoint loader)
        for i in range(2):                   
            if i == 1: input()                   # skip the line that prints which network it is
            if i == 0:
                inputs = input().split()
                time = float(inputs[-1])         # time to compute this Lipschitz constant is the final number on this line
            inputs = input().split()
            lip = float(inputs[2][7:-1])         # obtain the Lipschitz constant, but strip 7 characters "(Array(" and the final comma
            time_jitted = float(inputs[-1])      # (jitted) time is the final number on this line
            input()                              # skip line that prints the state space as a check
            res = input().split()
            first = float(res[0])
            time_first = float(res[2])
            time_lower = timeout
            while len(res) == 3:
                resold = res
                if time_lower == timeout and float(res[0]) < lip:
                    time_lower = float(res[2])
                res = input().split()
            better_timeout = 1 if (time_lower == timeout) else 0  # keep track of whether there was a timeout to reach the better lipschitz constant
            ub_final = float(resold[0])
            lb_final = float(resold[1])           
            input()                              # skip line that repeats the lip constant
            time_final = float(input())          # time taken to obtain exact lipschitz constant
            exact_timeout = 1 if (time_final >= timeout) else 0   # keep track of whether there was a timeout to reach the exact lipschitz constant
            if i == 0: policy += np.array([lip, time, time_jitted*1000, first, time_first, time_lower, ub_final, lb_final, time_final, better_timeout, exact_timeout])
            else: certificate += np.array([lip, np.nan, time_jitted*1000, first, time_first, time_lower, ub_final, lb_final, time_final, better_timeout, exact_timeout])
    assert nresults >= 1
    results_policy.append((policy/nresults).tolist())
    results_certificate.append((certificate/nresults).tolist())
    names.append(name)
    probabilities.append(probability)

print("""\\begin{table}
\\centering
\\begin{tabular}{lll|rrrr|rr}
\\toprule
model & $\\rho$ & network & $L_{\\text{Ours}}$ & $L_{\\text{LipBaB}, 1}$ &  $L_{\\text{LipBaB}}$ & lower & $t_{\\text{better}}$ & $t_{\\text{exact}}$ \\\\ \\midrule""")    
for res, name, p in zip(results_policy, names, probabilities):
    print_line(res, name, p, "$\pi$")
    
for res, name, p in zip(results_certificate, names, probabilities):
    print_line(res, name, p, "$V$")
print("""\\bottomrule
\\end{tabular}
\\medskip
\\end{table}""")
