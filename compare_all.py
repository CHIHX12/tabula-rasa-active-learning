#!/usr/bin/env python3
import json, glob, os
FRESH="results"; PRE="/home/cycheng/投稿/submission_code/precomputed_results"
def load(d,n):
    p=f"{d}/active_learning_v6_{n}_results.json"
    return json.load(open(p)) if os.path.exists(p) else None
configs=sorted(os.path.basename(x)[len('active_learning_v6_'):-len('_results.json')]
               for x in glob.glob(f"{PRE}/active_learning_v6_*_results.json"))
print(f"{'config':22s} {'status':6s} {'LCBDS_best Δ':>16s} {'break Δ':>10s} {'206 hit f/p':>12s}")
print("-"*78)
done=0
for c in configs:
    f=load(FRESH,c); p=load(PRE,c)
    if f is None:
        print(f"{c:22s} {'...':6s}"); continue
    done+=1
    db=f['final_best_mV']['lcb']-p['final_best_mV']['lcb']
    dbe=f['lcb_break']['mean']-p['lcb_break']['mean']
    fh=sum(1 for x in f['lcb_best_per_seed'] if x<=210)
    ph=sum(1 for x in p['lcb_best_per_seed'] if x<=210)
    print(f"{c:22s} {'DONE':6s} {f['final_best_mV']['lcb']:6.0f}/{p['final_best_mV']['lcb']:<6.0f}{db:+5.0f} {dbe:+10.1f} {fh:>5d}/{ph:<5d}")
print("-"*78)
print(f"{done}/{len(configs)} configs reproduced")
