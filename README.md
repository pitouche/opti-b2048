# b2048 — `-b/-ub 2048` (batch prefill/decode) au lieu de 512

**Verdict : GO — en prod** (depuis le 17/10 23:50, acheté contre ctx 200k→167k).

## Question

La prod tournait en **`-b 512` sans que personne l'ait choisi** : le scheduler
ollama (`server/sched.go:827 automaticGenerationBatch`) calcule `-b` tout seul,
et à 200k ctx + MTP + vision, le fit VRAM (≤ 60 % dispo pour un batch ≥ 2048)
**échouait** → cascade 2048→1024→512 → 512 = le pire des 4. Hypothèse : le chunk
de prefill sous-dimensionné pénalise le prefill (et, mesuré, le decode long ctx via
le workspace CUDA). Bench ladder 512/1024/2048/4096 de 16k à 196k, MTP off, même .so.

## Résultats (ab_confirm.py, 12 mesures valides, 3 reps ordre alterné, MTP off)

| ctx | A b512 decode | B b2048 decode | Δ | A b512 prefill | B b2048 prefill | Δ |
|---|---|---|---|---|---|---|
| 16 000 | 13,42 | 11,60 | **−13,6 %** | 394 | 680 | +72,6 % |
| 57 856 | 14,39 | 15,00 | +4,2 % | 488 | 650 | +33,2 % |
| 114 688 | 12,10 | 17,39 | **+43,7 %** | 550 | 620 | +12,7 % |
| 196 608 | 14,54 | 14,37 | −1,2 % | 437 | 464 | +6,2 % |

**Re-test sur .so T3 prod (d2df07b8), MTP OFF, 3 reps interleave** (`ab_batch_t3.py`) :

| ctx | A b2048 decode | B b512 decode | Δ | A b2048 prefill | B b512 prefill | Δ |
|---|---|---|---|---|---|---|
| 114 688 | 21,02 | 19,19 | **+9,5 %** | 623,1 | 559,0 | +11,5 % |
| 196 608 | 18,18 | 14,47 | **+25,6 %** | 465,7 | 426,6 | +9,2 % |

Lecture : **b2048 gagne le decode long ctx** (le point d'usage) et le prefill
partout ; **b512 gagne le decode court ctx** (16 k : −13,6 %). Le decode court
n'est pas le point d'usage (c'est le long ctx qui coûte) → le trade est favorable.
Le « pourquoi » du gain decode long ctx (workspace CUDA / fit) est documenté dans
`results/b2048_REEVAL_datas.md` §2-3.

## Le prix (le design même de la piste)

Le bench 196k **sans MTP** passait le fit à 2048 (30,7 GB) ; le runner 200k
**MTP+vision** n'avait pas les ~0,8 GB de marge. Résolution du 17/10 (faite à la
main) : **descendre le ctx 200k → 167k** (175k testé et échoué, 167k passe avec
60 MB de free) pour libérer la place du workspace b2048. **Le b2048 a toujours été
acheté contre ~16,5 % de contexte** — c'est son design, pas un bordélate.

## Contenu

```
results/  # b2048_REEVAL_datas.md (le doc de référence, 253 lignes),
          # ab_batch_t3_summary.md + ab_batch_t3_results.jsonl (re-test .so T3),
          # ab_batch_t3_curve_summary.md + ab_batch_t3_curve_results.jsonl (courbe 4 batches)
bench/  # ab_batch_t3.py, ab_batch_t3_curve.py, b2048_t3_fig.py, nso_orchestrator_b2048.py
```

Pas de `.so` : c'est un paramètre de lancement (`num_batch`), pas un kernel.
Le .so de référence du re-test est `d2df07b8` (T3 prod de l'époque).

## Re-validation

`python3 bench/bench/supervisor_bench.py run bench/ab_batch_t3.py --detach`
→ A-b2048 / B-b512, 2 ctx × 3 reps, ~30 min. Comparer à
`results/ab_batch_t3_results.jsonl` (A 21,02/18,18 ; B 19,19/14,47 decode).
Critère : gain decode long ctx > 5 % (ici +9,5/+25,6 %). Le ctx court est
**attendu en perte** — c'est le trade accepté (documenté, pas un bug).

## Caveats

- Le fit dépend du ctx + du .so + du KV : **si le ctx change (ou le .so change
  assez pour changer le workspace), re-valider le fit** (175k échoue, 167k passe —
  marge de 60 MB seulement).
- Le decode court ctx est **réellement plus lent en b2048** (−13,6 % @16 k). Si
  le profil d'usage change vers du court ctx, c'est un argument pour ré-ouvrir.
- Le mécanisme exact du gain decode long ctx (workspace CUDA) est partiellement
  documenté ; le chiffre e2e (mesuré) fait foi même si le mécanisme est incomplet.
