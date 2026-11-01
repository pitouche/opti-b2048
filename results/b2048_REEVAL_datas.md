# Réévaluation b2048 — datas constituées à l'époque (consolidé 22/10)

Source : `BILAN_SESSION_20261017.md` (session du levier batch, déploiement),
`ab_batch_results.jsonl` + `ab_batch_summary.md` (sweep 4 batches × 8 points, 17/10 20:14 UTC),
`ab_confirm_results.jsonl` (12 mesures valides, 3 reps ordre alterné),
`night_bench_summary.md` + `night_bench_depeouille.md` (EXP2-BATCH, 18/10),
`deploy_175k_*/ollama-v100.service.bak` (état prod avant changement),
`deploy_batch2048.sh` (mécanisme de déploiement).

## 0. ORIGINE DE LA PISTE b2048 (reconstituée 22/10)

**En une ligne : la piste b2048 est née le 17/10 par élimination — 3ᵉ levier d'une
session « quick wins sans rebuild », après l'échec de deux leviers, et elle était
forcée par une observation scheduler, pas par un choix de config.**

### 0.1 L'élimination du 17/10 (chat « #1 Optimisations V100 »)

| heure (log superviseur) | levier testé | résultat |
|---|---|---|
| matin | patch `42d9581` (MMVQ K-quant nwarps 4→2) | **rejeté** (−1 à −25 %, bruit) — `ab_mmvq2_CONSOLIDE.md` |
| 13:28 / 14:32 | env VOLTA absents de la prod (`Q5_X4`, `Q8_FATTN_TC`) | **neutres** (±2 %) — `ab_env_BILAN.md` |
| **18:19** | **`-b/-ub` 512→4096 (`ab_batch.py`)** | **GAGNANT → confirmé 20:40 (`ab_confirm.py`) → déployé 23:50** |

Preuve du ré-usage du fichier : le docstring de `ab_batch.py` est **encore** celui
du bench env (legs A-base/B-q5x4/C-q5x4+q8fa) — le fichier a été réutilisé tel quel
pour l'AB batch (legs A-b512/B-b1024/C-b2048/D-b4096) après l'échec des env, sans
mettre le docstring à jour.

### 0.2 Pourquoi le batch s'imposait : observation scheduler, pas config

- La prod tournait en **`-b 512` sans que personne l'ait choisi** : le scheduler
  ollama (`server/sched.go:827 automaticGenerationBatch`) calcule `-b` tout seul
  (détail mécanique : §3).
- À 200k ctx + MTP + vision, le fit VRAM (≤ 60 % de la dispo pour un batch ≥ 2048)
  **échouait** → cascade 2048→1024→512 → **512 = le pire des 4** (table §2).
- Hypothèse : le chunk de prefill sous-dimensionné pénalise le prefill (et, mesuré
  après coup, le decode long ctx via le workspace CUDA) → bench ladder
  512/1024/2048/4096 de 16k à 196k, MTP off, même .so.

### 0.3 Le prix (le design même de la piste)

Le bench 196k **sans MTP** passait le fit à 2048 (30,7 GB) ; le runner 200k
**MTP+vision** n'avait pas les ~0,8 GB de marge. Résolution du 17/10 23:50 (faite à
la main par guile) : **descendre le ctx 200k → 167k** (175k testé et échoué, 167k
passe avec 60 MB de free) pour libérer la place du workspace b2048. Le b2048 a
toujours été **acheté contre ~16,5 % de contexte** — c'est son design, pas un
bordélate.

Trace de déploiement : `deploy_batch2048.sh` (fixe `num_batch` dans le manifest,
`usesAutomaticNumBatch()=false`) figure dans les fichiers de session comme
« abandonné » au profit du combo ctx+b2048 via `deploy_175k.sh` (dérivé) ; l'état
prod final = manifest `"num_batch":2048` + unit `-c 167000` (mécanisme d'inversion :
§5).

## 1. Le trade-off exact

| | Avant (prod pré-17/10) | Après (prod depuis 17/10 23:50) |
|---|---|---|
| ctx | `OLLAMA_CONTEXT_LENGTH=200000` (backup manifest/unit le prouvent) | `OLLAMA_CONTEXT_LENGTH=167000` |
| batch | `-b 512 -ub 512` (choisi par le scheduler ollama, voir §3) | `-b 2048 -ub 2048` (fixés dans le manifest via `"num_batch":2048`) |
| VRAM runner | 32.35 GB → fit test échouait | 32 436/32 768 MiB, marge ~60 MB (Grafana) |
| Δ ctx | — | **−16,5 %** (200k → 167k) |

**175k a été testé et a échoué** (allocs CUDA, granularité ; la KV 175k = 5 811 MiB
théoriques mais l'alloc passe à 170k, pas 175k). 167k passe, 170k aussi en alloc
mais c'est 167k qui a été retenu. 0 eviction depuis le démarrage du runner 167k.

## 2. Le gain mesuré (MTP OFF, .so a4b6f440, interleave, 3 reps, variance < 1 %)

**Decode (t/s)** — le gain principal, pas seulement le prefill :
| contexte | b512 | b2048 | Δ |
|---:|---:|---:|---:|
| 114 688 (régime prod) | 12.08 (12.12/12.08/11.99) | 17.39 (17.32/17.46/17.39) | **+44 %** |
| 196 608 | 10.46 (10.43/10.46/10.63) | 14.13 (14.13/14.13/14.13) | **+35 %** |
| 16–32 k | 23–25 | idem | 0 % (stable, attendu) |

**Prefill (t/s)** :
| contexte | b512 | b1024 | b2048 | b4096 |
|---:|---:|---:|---:|---:|
| 114 688 | 551 | 606 | 620 | 636 |
| 196 608 | 420 | 452 | 464 | 478 |
| 16 k (cold) | 874 | 1 003 | 1 047 | 1 068 |
| 32–98 k | 684–809 | +8–9 % | +10–11 % | +11–13 % |

→ b512→b2048 = **prefill +10 à +12 %** sur toute la plage ; b2048→b4096 ne rajoute
que +2–3 % (b4096 a été écarté : +1.2 GB VRAM pic sans gain proportionnel).

Note de cohérence : le gain decode long-ctx (+35-44 %) mesuré MTP-off est plus gros
que le gain prefill — c'est ce qui fait le poids du levier au total, pas « juste un
peu de prefill ». (Mécanisme : le scheduler choisissait 512, ce qui pénalise aussi
le decode long ctx dans les conditions du bench.)

## 3. POURQUOI b2048 exigeait de baisser le ctx (cœur du trade-off)

Scheduler ollama `server/sched.go:827 automaticGenerationBatch` :
- ctx > 32k → démarre à 2048, puis `generationBatchFits()` exige VRAM prédite
  ≤ 60 % de la dispo ; sinon cascade 2048 → 1024 → 512.
- À **200k + MTP + vision**, le runner montait à 32.35 GB sur V100 32 GB →
  le fit **échouait** → retombée **512, le pire des 4** (table §2).
- À 196k sans MTP, le fit passait à 2048 (30.7 GB) → le gain était mesurable au
  bench, mais le runner 200k réel n'avait pas les ~0.8 GB de marge.
- D'où : descendre ctx à 167k pour libérer la marge → b2048 passe le fit.

Le bench a donc été fait MTP-off (b512/b2048 comparables sans MTP) ; en prod
(MTP ON) le cycle draft/verify change les ratios → **item restant du 17/10** :
confirmer en live Grafana (t/s live pré-deploy ~26-30 au régime 114k MTP-on ;
post-deploy « t/s live ~30 » noté le 18/10 00:30, signal faible, jamais
consigné proprement en A/B MTP-on).

## 4. EXP2-BATCH night bench 18/10 (b4096/b8192 vs b2048, MTP off, 2 reps)

| ctx | A-b2048 dec | B-b4096 dec | prefill 2048/4096 |
|---:|---:|---:|---:|
| 114 688 | 17.69 | 17.68 (−0.1 %) | 619.8 / 636.4 |
| 196 608 | 14.13 | 14.13 (+0.0 %) | 343.6 / 354.9 |

VRAM pic : b2048 = 30 988 MiB, b4096 = 32 144 MiB, b8192 = 32 477 MiB (C non mesuré,
déchus). → **b2048 = point d'quilibre** ; au-delà, c'est de la VRAM pour 0 % de gain.

## 5. Mécanisme de déploiement (à inverser si on revient en arrière)

- `deploy_batch2048.sh` : écrit un blob params `{"draft_num_predict":3,"num_batch":2048}`
  dans le manifest ollama (`usesAutomaticNumBatch()=false` → `-b 2048 -ub 2048` fixes).
  Le blob params actuel est le sha256 de cette JSON.
- Backup des manifests avant changement : `deploy_175k_20261017-231437/` et
  `-231957/` (params blob `940a4523…`, size 24 = `{"draft_num_predict":3}` seul,
  ctx 200000 dans l'unit bak).
- Pour revenir à 200k+b512 : restaurer l'unit bak (ctx 200000) + restaurer le
  manifest bak (params sans num_batch → scheduler repasse en automatique) +
  reload/restart. Le scheduler re-choisirait 512 (fit échoué à 200k, §3) — à
  confirmer par un `ps` du runner après redémarrage.

## 6. Points d'attention pour la réévaluation

1. **Le coût ctx (-16,5 %) n'a jamais été chiffré en applicatif** : personne n'a
   mesuré l'impact réel du ctx 200k→167k sur les usages (évictions de cache,
   re-prefills, shift `--context-shift` qui gère le débordement depuis le 17/10).
   C'est le trou dans les données si on veut trancher.
2. **Le gain decode MTP-on en live n'a jamais été consigné proprement** (item 1 du
   §6 du BILAN_SESSION_20261017, toujours ouvert) : les benchs sont MTP-off ;
   avec MTP ON le decode 114k passe par le cycle draft/verify (T3 inclus depuis le
   18/10), et le poids réel du b2048 sur le t/s user final est inconnu.
3. **Marge VRAM ~60 MB** : « N'AJOUTER AUCUN autre buffer sur ce runner sans
   redescendre le ctx » (piège n°3 du 17/10). Tout repli vers 200k re-bloque b2048.
4. Si la réévaluation penche pour « retour 200k + b512 » : on perd +44 % decode
   @114k / +35 % @196k (bench MTP-off) et +10-12 % prefill ; si elle penche pour
   « garder 167k + b2048 » : le point faible est le ctx perdu et la marge 60 MB.

## 7. RE-TEST 22/10 sur le .so T3 prod (d2df07b8) — le re-test demandé

`ab_batch_t3.py` (mêmes points/protocole que l'ab_confirm du 17/10 : 2 ctx × 2 legs
× 3 reps, ordre alterné, serveur frais par leg, port 8917, KV q8_0, FA auto, H2 OFF,
Q6_W4R4=1, MTP OFF volontairement). A-b2048 = prod actuelle, B-b512 = T3 sans b2048
(ce que le scheduler choisirait à 200k). Durée 69 min, rc=0, 12/12 ok.
Résultats : `ab_batch_t3_results.jsonl` + `ab_batch_t3_summary.md`.

**Decode (t/s), médianes :**
| contexte | A b2048 (T3) | B b512 (T3) | Δ b2048 vs b512 | gain 17/10 (so ancien) |
|---:|---:|---:|---:|---:|
| 114 688 | 21.02 (21.02/21.02/21.03) | 19.19 (20.97/19.19/18.87) | **+9.5 %** (reps +0.3/+9.5/+11.4) | +44 % |
| 196 608 | 18.18 (18.18/18.18/18.19) | 14.47 (14.47/14.38/14.81) | **+25.6 %** (reps +25.7/+26.4/+22.8) | +35 % |

**Prefill (t/s), médianes :**
| contexte | A b2048 | B b512 | Δ | gain 17/10 |
|---:|---:|---:|---:|---:|
| 114 688 | 623.1 | 559.0 | +11.5 % | +12 % |
| 196 608 | 465.7 | 426.6 | +9.2 % | +11 % |

**Ce que T3 a changé dans le trade-off (comparaison des 2 benchs, MTP off, même protocole) :**
- T3 a levé les DEUX jambes, surtout b512 : decode b512 +59 % @114k (12.08→19.19)
et +38 % @196k (10.46→14.47) ; b2048 +21 % / +29 %. Le FA decode (rows≤8) était le
bouteille, T3 l'a compressé, donc le poids relatif du batch a fondu.
- **Le gain b2048 decode est passé de +44/+35 % (17/10) à +9.5/+25.6 % (22/10, .so T3)**.
  @114k il est petit et bruité (reps +0.3…+11.4 %, variance B ~±5 %) ; @196k il reste
  net et stable (+22.8…+26.4 %).
- Prefill : inchangé, +9 à +12 % partout — c'est le seul gain b2048 constant.
- Le point 196 608 = exactement le régime du ctx 200k perdu : avec b512, un user qui
  irait à ~196k tournerait à **14.5 t/s** au lieu de **18.2 t/s** avec b2048 (−20 %).

**Lecture pour la décision :**
- Coût b2048 actuel : −16.5 % de ctx (167k vs 200k), marge VRAM 60 MB.
- Ce qu'il rapporte aujourdhui (.so T3) : prefill +9-12 % (toujours), decode +9.5 %
  @114k (marge, bruité) / +25.6 % @196k (solide, mais régime inaccessible sans les
  33k de ctx rendus), et il maintient la marge 60 MB qui autorise T3+MTP+vision sur
  une seule V100.
- L'argument d'origine du 17/10 (« +44 % decode à 114k ») n'est plus d'actualité : c'était
  le gain T3 manquant qui se logeait dans la comparaison. Le b2048 d'aujourd'hui vaut
  surtout : prefill +10 %, decode long-ctx >180k +25 %, et la VRAM libérée.

## 7.5. COURBE COMPLÈTE b2048-sur-T3, 16k → 196k (22/10, run 15:23 UTC)

`ab_batch_t3_curve.py` — même protocole que `ab_batch_t3.py` (2 reps, ordre alterné,
serveur frais par leg, MTP OFF, H2 OFF), points 16 384 / 32 768 / 65 536 / 98 304 /
131 072 / 163 840 / 196 608 × 2 legs. 114 688 pas re-mesuré (déjà 3 reps dans
`ab_batch_t3.py`). 28/28 mesures ok, rc=0, service restauré actif. Durée ~95 min.
Résultats : `ab_batch_t3_curve_results.jsonl` + `ab_batch_t3_curve_summary.md`.

**La courbe « delta b2048-sur-T3 par contexte » (médianes, Δ = A-b2048 vs B-b512) :**

| ctx | decode A | decode B | Δ decode | prefill A | prefill B | Δ prefill | wall A (s) | wall B (s) | Δ wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 384 | 25.52 | 25.55 | −0.1 % | 1053 | 884 | **+19.1 %** | 17.8 | 20.7 | −2.9 s |
| 32 768 | 24.57 | 24.56 | 0.0 % | 951 | 814 | **+16.9 %** | 36.3 | 42.0 | −5.7 s |
| 65 536 | 23.04 | 23.03 | 0.0 % | 786 | 690 | **+14.0 %** | 84.3 | 95.7 | −11.4 s |
| 98 304 | 21.66 | 18.51 | +17.0 % | 668 | 597 | **+12.0 %** | 146.6 | 164.9 | −18.3 s |
| 114 688* | 21.02 | 19.19 | +9.5 % | 623 | 559 | **+11.5 %** | 182.5 | 203.4 | −20.9 s |
| 131 072 | 20.38 | 14.15 | +44.0 % | 584 | 526 | **+11.1 %** | 222.2 | 248.0 | −25.8 s |
| 163 840 | 19.16 | 13.77 | +39.1 % | 518 | 471 | **+10.0 %** | 312.1 | 344.9 | −32.8 s |
| 196 608 | 18.19 | 13.52 | +34.5 % | 466 | 426 | **+9.3 %** | 415.4 | 455.3 | −39.9 s |

\* 114 688 = ab_batch_t3.py (3 reps), le seul point du run de 22/10 à 3 reps.

**Lecture de la courbe (c'est elle la réponse à « combien b2048 apporte on top de T3 ») :**
1. **Decode : plat à 0 % jusqu'à 65k, puis rupture brutale entre 65k et 131k**
   (0 % → +44 %), plateau ~+35-45 % au-dessus. La jambe B-b512 se dégrade nettement
   avec la taille KV (13.5-14 t/s @131-196k) tandis que A-b2048 descend en douceur
   (18-20 t/s) : ce n'est pas un gain actif, c'est **B qui s'effondre** — le workspace
   CUDA de -b 512 limite le FA sur KV longue (T3 consomme le workspace autrement que
   le FA stock : split-KV intra-CTA vs K-parallel inter-CTAs). Seuil ≈ 98k (1er rep
   +17 %, 2e +35 % — déjà actif, 1 rep bruité).
   ⚠️ Le +9.5 % de 114 688 (3 reps, ab_batch_t3.py) est un point isolé sous le
   plateau 131k — à replier vers la courbe 98k→131k plutôt qu'à prendre au pied de la lettre.
2. **Prefill : le vrai levier constant de b2048, +9 à +19 %, décroissant avec le ctx**
   (plus gros sur les petites requêtes). T3 ne touche pas le prefill → ce gain est
   100 % « batch ».
3. **Wall (ce que le user attend) : −2.9 s @16k → −40 s @196k**. En % du wall :
   −14 % @16k, −13 % @32k, −12 % @65k, −11 % @98k, −10 % @115k, −10 % @131k,
   −9.5 % @164k, −8.8 % @196k. À court/moyen ctx c'est 100 % du prefill ; au-delà de
   98k le decode long rejoint et le gain wall tient à ~10 %.
4. **Comparaison avec l'ancien .so (ab_batch 17/10, sans T3)** : sur l'ancien .so le
   gain decode était déjà +35-51 % @98-196k → **T3 n'a PAS absorbé le gain b2048 au
   long ctx** (contradiction avec la lecture 114k du re-test 2 points) ; par contre à
   court ctx (16-32k) c'était déjà 0 % et reste 0 %. Le re-test 2 points avait donc
   surestimé l'effacement T3 à 114k (3 reps mais un seul point sous le seuil).

**Réponse consolidée « combien b2048 on top of T3 » :**
- ctx ≤ 65k : **prefill +14-19 % uniquement, decode 0** → wall −11 à −14 %.
- 98k ≤ ctx ≤ 167k (le regime prod réel) : **prefill +10-12 % + decode +17-44 %** → wall ~−10 %.
- > 167k (inaccessible, le ctx perdu) : prefill +9-10 % + decode +35-45 % → wall −9 à −10 %.
- Le b2048 d'aujourd'hui est donc surtout un levier **prefill constant partout** (+ un
  levier decode >98k), pas un levier decode court ctx. Décision : **garder b2048** —
  le prefill +10-19 % est payant dès 16k, et le drop à 167k n'ampute que le régime
  >167k qui ne se paie pas le +35 % decode qu'on y perdrait avec b512.

## Fichiers bruts
- `ab_batch_results.jsonl` (32 lignes, 8 points × 4 batches)
- `ab_confirm_results.jsonl` (12 lignes valides, b512 vs b2048, .so a4b6f440)
- `ab_batch_summary.md`, `ab_confirm.py` (table Δ par rep)
- `night_bench_depeouille.md` (EXP2-BATCH)
- `BILAN_SESSION_20261017.md` (story complète du 17/10)
- `deploy_175k_20261017-231437/{manifest.latest.bak,ollama-v100.service.bak}` (état 200k)
- `ab_batch_t3.py` / `ab_batch_t3_results.jsonl` / `ab_batch_t3_summary.md` (re-test 22/10, .so T3, 2 ctx × 3 reps)
- `ab_batch_t3_curve.py` / `ab_batch_t3_curve_results.jsonl` / `ab_batch_t3_curve_summary.md` (courbe 16k→196k, 22/10, 7 ctx × 2 reps)
