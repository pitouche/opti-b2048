#!/usr/bin/env python3
"""
b2048_t3_fig.py — 2 graphiques (decode / prefill) : 4 jambes b512/b2048 × sans-T3/T3.

Données (MTP OFF, même protocole, serveur frais par leg) :
- .so ancien (a4b6f440, sans T3, 17/10) : ab_batch_results.jsonl (legs A-b512, C-b2048)
- .so T3 (d2df07b8, 22/10) : ab_batch_t3_results.jsonl (114688/196608, 3 reps)
                             + ab_batch_t3_curve_results.jsonl (7 pts, 2 reps)
114688 et 196608 côté T3 : médiane sur les 3 reps d'ab_batch_t3 (point 196608
du run courbe ignoré pour éviter le double comptage).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "b2048_t3_curves.png")


def med(vals):
    vals = sorted(v for v in vals if v)
    return vals[len(vals) // 2] if vals else None


# ---- .so ancien (sans T3) ----
old = {}
for l in open(os.path.join(D, "ab_batch_results.jsonl")):
    r = json.loads(l)
    old.setdefault(r["leg"], {}).setdefault(r["ctx"], []).append(r)
old_b512 = {c: {"pre": med([x.get("prompt_tps") for x in v]),
                "dec": med([x.get("eval_tps") for x in v])} for c, v in old["A-b512"].items()}
old_b2048 = {c: {"pre": med([x.get("prompt_tps") for x in v]),
                 "dec": med([x.get("eval_tps") for x in v])} for c, v in old["C-b2048"].items()}

# ---- .so T3 : courbe (2 reps) + re-test 2 pts (3 reps pour 114688/196608) ----
t3 = {}
for f, pts in (("ab_batch_t3_curve_results.jsonl", None), ("ab_batch_t3_results.jsonl", {114688, 196608})):
    for l in open(os.path.join(D, f)):
        r = json.loads(l)
        if pts is not None and r["ctx"] not in pts:
            continue
        t3.setdefault(r["leg"], {}).setdefault(r["ctx"], []).append(r)
t3_b512 = {c: {"pre": med([x.get("prompt_tps") for x in v]),
               "dec": med([x.get("eval_tps") for x in v])} for c, v in t3["B-b512"].items()}
t3_b2048 = {c: {"pre": med([x.get("prompt_tps") for x in v]),
                "dec": med([x.get("eval_tps") for x in v])} for c, v in t3["A-b2048"].items()}

CTX_ALL = sorted(set(old_b512) | set(old_b2048) | set(t3_b512) | set(t3_b2048))


def plot_panel(ax, key, title, ylim):
    series = [
        ("no opti (b512, .so ancien)", CTX_ALL, old_b512, key, "#888888", "--", 1.6, 4),
        ("opti b2048 (.so ancien)", CTX_ALL, old_b2048, key, "#e07b00", "--", 1.8, 5),
        ("opti T3 seul (b512)", CTX_ALL, t3_b512, key, "#2b7bba", "-", 2.0, 6),
        ("opti b2048 + T3  ← PROD", CTX_ALL, t3_b2048, key, "#1e8449", "-", 3.0, 7),
    ]
    for label, ctxs, data, k, color, ls, lw, ms in series:
        xs = [c / 1000.0 for c in ctxs if c in data and data[c][k]]
        ys = [data[c][k] for c in ctxs if c in data and data[c][k]]
        ax.plot(xs, ys, label=label, color=color, ls=ls, lw=lw, marker="o",
                markersize=ms - 2, markerfacecolor="white", markeredgewidth=1.5, zorder=3)
    ax.axvline(167, color="#1e8449", ls=":", lw=1.2, alpha=0.6, zorder=1)
    ax.text(168.5, ylim[1] * 0.97, "ctx prod\n167 k", fontsize=8, color="#1e8449", alpha=0.8, va="top")
    ax.axvline(114.7, color="#555555", ls=":", lw=1.0, alpha=0.4, zorder=1)
    ax.text(115.7, ylim[1] * 0.97, "régime\n115 k", fontsize=8, color="#555555", alpha=0.7, va="top")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("taille du contexte (k tokens)", fontsize=10)
    ax.set_ylabel("tok/s", fontsize=10)
    ax.set_xlim(12, 205)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, lw=0.7)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
plot_panel(ax1, "dec", "DECODE (t/s) — MTP OFF", (8, 28))
plot_panel(ax2, "pre", "PREFILL (t/s) — MTP OFF", (350, 1150))

fig.suptitle(
    "b2048 × T3 — 4 jambes, Qwen3.8-27B Q6 / KV q8_0 / FA auto / H2 OFF (V100, .so dlopen)\n"
    ".so ancien = a4b6f440 (17/10, sans T3) · .so T3 = d2df07b8 prod (22/10) · "
    "note : point 16 k .so ancien mesuré cold (premier chargement)",
    fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("écrit :", OUT)

# petite table de contrôle
print(f"{'ctx':>8} | {'no opti dec':>11} | {'b2048 dec':>9} | {'T3 dec':>7} | {'b2048+T3 dec':>12} | "
      f"{'no opti pre':>11} | {'b2048 pre':>9} | {'T3 pre':>7} | {'b2048+T3 pre':>12}")
for c in CTX_ALL:
    g = lambda d, k: d.get(c, {}).get(k)
    print(f"{c:>8} | {g(old_b512,'dec') or '-':>11} | {g(old_b2048,'dec') or '-':>9} | "
          f"{g(t3_b512,'dec') or '-':>7} | {g(t3_b2048,'dec') or '-':>12} | "
          f"{g(old_b512,'pre') or '-':>11} | {g(old_b2048,'pre') or '-':>9} | "
          f"{g(t3_b512,'pre') or '-':>7} | {g(t3_b2048,'pre') or '-':>12}")
