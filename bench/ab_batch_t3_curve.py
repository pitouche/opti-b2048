#!/usr/bin/env python3
"""
ab_batch_t3_curve.py — COURBE COMPLÈTE b2048 sur T3, MTP OFF.

Courbe b2048-sur-T3 de 16k à 196k (le trou du re-test 22/10 qui ne couvrait
que 114k/196k). Même protocole que ab_batch_t3.py :
  A-b2048 : T3 + -b 2048 -ub 2048 (prod actuelle)
  B-b512  : T3 + -b 512 -ub 512 (ce que le scheduler choisirait à 200k)
mêmes points/geometry que l'ab_batch du 17/10 (comparabilité 2 .so),
2 reps (coût), ordre alterné, serveur frais par leg.

114688 n'est PAS re-mesuré : il est déjà couvert à 3 reps par ab_batch_t3.py.

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run ab_batch_t3_curve.py --detach
"""
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from gpu_uuid_select import v100_cuda_ordinal, v100_smi_index  # UUID GPU (jamais d'index en dur — voir GPU_UUID_NSYS_CUPTI_20261030.md)

D = os.path.dirname(os.path.abspath(__file__))

LLAMA_SERVER = "/home/guile/llama-experiments/ollama-v100/lib/ollama/llama-server"
GGUF = (
    "/usr/share/ollama/.ollama/models/blobs/"
    "sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
MMPROJ = (
    "/usr/share/ollama/.ollama/models/blobs/"
    "sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e")

PORT = 8917
SERVICE = "ollama-v100"
OUT_JSONL = os.path.join(D, "ab_batch_t3_curve_results.jsonl")
OUT_SUMMARY = os.path.join(D, "ab_batch_t3_curve_summary.md")
OUT_SERVER_LOG = os.path.join(D, "ab_batch_t3_curve_server.log")

CUDA_LIBS = [
    "/home/guile/llama-experiments/ollama-v100/lib/ollama",
    "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/lib64",
    "/home/guile/llama-experiments/tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib",
]
SO_PROD = "/home/guile/llama-experiments/ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so"
SO_EXPECTED_PREFIX = "d2df07b8"   # .so T3 prod — garde anti-dérive

# prod MTP-OFF : Q6_W4R4=1, H2 OFF (FATTN_TC=0), T3 ON (défaut).
ENV_BASE = {
    "GGML_CUDA_VOLTA_Q6_W4R4": "1",
    "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",
    "LLAMA_ARG_FIT_TARGET": "1024",
}
LEGS = [
    ("A-b2048", dict(ENV_BASE, _B="2048")),
    ("B-b512",  dict(ENV_BASE, _B="512")),
]

# courbe 16k -> 196k (114688 déjà couvert 3 reps par ab_batch_t3.py)
POINTS = [
    (16384, 64),
    (32768, 64),
    (65536, 64),
    (98304, 64),
    (131072, 64),
    (163840, 64),
    (196608, 64),
]
REPS = 2
MAX_CTX = 196672
TIMEOUT_S = 900
GPU_SMI_IDX = v100_smi_index()  # résolu par UUID (ne pas hardcoder)
GPU_CUDA_IDX = v100_cuda_ordinal()  # résolu par UUID (ne pas hardcoder)
FILL = ("La tour de contrôle surveille le passage des avions au crépuscule "
        "pendant que le chef de piste organise les départs de la soirée. ")

_state = {}
_restored = False


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def svc_state():
    return run(["systemctl", "is-active", SERVICE]).stdout.strip()


def vram_mib():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1


def v100_proc_gone():
    r = run(["nvidia-smi", f"--id={GPU_SMI_IDX}",
             "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    for line in r.stdout.splitlines():
        if line.strip():
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip().replace("MiB", "").replace(" ", "").isdigit():
                if int(parts[1].strip().split()[0]) > 8000:
                    return False
    return True


def restore_service():
    global _restored
    try:
        if svc_state() != "active":
            log(f"  · restauration : systemctl start {SERVICE}")
            run(["sudo", "systemctl", "start", SERVICE], timeout=120)
        for _ in range(90):
            if svc_state() == "active":
                break
            time.sleep(2)
        _restored = svc_state() == "active"
        log(f"  · service {SERVICE} = {svc_state()}")
    except Exception as e:
        log(f"  · ERREUR restauration : {e}")
        _restored = False


def on_sig(sig, frm):
    restore_service()
    sys.exit(128 + sig)


def stop_and_free():
    log(f"  · stop {SERVICE}")
    run(["sudo", "systemctl", "stop", SERVICE], timeout=120)
    t0 = time.time()
    while time.time() - t0 < 180:
        if vram_mib() < 1024 and v100_proc_gone():
            log(f"  · V100 libérée ({vram_mib()} MiB) en {time.time()-t0:.0f}s")
            return True
        time.sleep(3)
    log(f"  · ATTENTION V100 pas totalement libérée (VRAM={vram_mib()} MiB)")
    return False


def build_env(extra):
    env = dict(os.environ)
    for d in CUDA_LIBS:
        if os.path.isdir(d) and d not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = d + ":" + env.get("LD_LIBRARY_PATH", "")
    env.update(extra)
    env["GGML_BACKEND_PATH"] = SO_PROD
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_CUDA_IDX)
    return env


def server_args(batch):
    return [
        "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
        "--no-webui", "--offline",
        "-c", str(MAX_CTX), "-np", "1",
        "-b", batch, "-ub", batch,
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
        "--mmproj", MMPROJ,
        # MTP OFF : pas de --spec-type
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--flash-attn", "auto",
    ]


def port_free(port):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port)); s.close(); return True
    except OSError:
        s.close(); return False


def http_json(url, data=None, timeout=10):
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def start_server(extra):
    batch = extra.get("_B", "512")
    env = {k: v for k, v in extra.items() if k != "_B"}
    if vram_mib() > 8192:
        raise RuntimeError(f"V100 pas vide (VRAM={vram_mib()} MiB) — refus de démarrer")
    env = build_env(env)
    lf = open(OUT_SERVER_LOG, "a")
    proc = subprocess.Popen([LLAMA_SERVER] + server_args(batch), stdout=lf,
                            stderr=subprocess.STDOUT, env=env)
    _state["proc"] = proc
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server mort au démarrage (exit {proc.returncode})")
        try:
            http_json(f"http://127.0.0.1:{PORT}/health", timeout=3); break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("llama-server : pas de /health en 600 s")
    time.sleep(3)
    if vram_mib() < 1024:
        raise RuntimeError(f"ABORT : VRAM={vram_mib()} MiB — model en CPU")
    d = http_json(f"http://127.0.0.1:{PORT}/v1/models", timeout=10)
    _state["model_id"] = d["data"][0]["id"]
    log(f"  · server prêt ({time.time()-t0:.0f}s, VRAM {vram_mib()} MiB) -b {batch}")
    return proc


def stop_server():
    p = _state.get("proc")
    if p and p.poll() is None:
        p.terminate()
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(timeout=10)
    time.sleep(8)
    for _ in range(60):
        if vram_mib() < 4096: break
        time.sleep(3)
    log(f"  · server stop, VRAM {vram_mib()} MiB")


def prompt_text(ctx_target_tokens):
    n_words = ctx_target_tokens * 4 // 5
    words = FILL.split() * max(1, n_words // FILL.count(" ") + 1)
    return " ".join(words[: n_words])


def parse_point(logf, since_pos):
    with open(logf, "r", errors="replace") as f:
        data = f.read()[since_pos:]
    out = {}
    m = re.search(r"prompt eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["prompt_tps"] = round(n / (ms / 1000.0), 1) if ms else 0
        out["prompt_tokens"] = n
    m = re.search(r"(?<!prompt )eval time =\s+([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", data)
    if m:
        ms, n = float(m.group(1)), int(m.group(2))
        out["eval_tps"] = round(n / (ms / 1000.0), 2) if ms else 0
        out["eval_tokens"] = n
        out["eval_ms"] = round(ms, 1)
    return out


def run_point(ctx_target, n_gen, since_pos):
    text = prompt_text(ctx_target)
    data = {
        "model": _state["model_id"], "stream": False,
        "messages": [{"role": "user", "content": text + "\nRéponds en une phrase."}],
        "max_tokens": n_gen, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
    }
    t0 = time.time()
    try:
        r = http_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", data, timeout=TIMEOUT_S)
        u = (r or {}).get("usage", {})
        api = {"prompt_tokens_api": u.get("prompt_tokens"),
               "gen_tokens_api": u.get("completion_tokens"),
               "wall_s": round(time.time() - t0, 1)}
    except Exception as e:
        api = {"error": str(e)}
    return api, parse_point(OUT_SERVER_LOG, since_pos)


def log_size():
    try: return os.path.getsize(OUT_SERVER_LOG)
    except OSError: return 0


def med(vals):
    vals = sorted(v for v in vals if v)
    return vals[len(vals) // 2] if vals else None


def so_prefix():
    h = hashlib.md5(open(SO_PROD, "rb").read()).hexdigest()[:8]
    return h


def write_summary(allrows):
    lines = ["# COURBE b2048-sur-T3, 16k → 196k — .so T3 prod (d2df07b8), MTP OFF", "",
             f"date : {time.strftime('%Y-%m-%d %H:%M UTC')}",
             f"service restauré : {'OUI' if _restored else 'NON'}",
             f".so prod : md5 prefix {so_prefix()} (attendu {SO_EXPECTED_PREFIX})",
             "",
             "A-b2048 = prod actuelle (T3 + -b 2048) · B-b512 = T3 sans b2048.",
             "2 reps, ordre alterné, serveur frais par leg, port 8917, CVD=0, KV q8_0, FA auto, H2 OFF.",
             "114688 hors run : déjà couvert à 3 reps par ab_batch_t3.py (22/10).",
             "",
             "## DECODE (t/s) par leg (Δ = A-b2048 vs B-b512)",
             "| contexte | rep | A b2048 | B b512 | Δ |",
             "|---|---|---|---|---|"]
    for ctx, ng in POINTS:
        for rep in range(1, REPS + 1):
            a = next((r for r in allrows if r["ctx"] == ctx and r["leg"] == "A-b2048" and r.get("rep") == rep), None)
            b = next((r for r in allrows if r["ctx"] == ctx and r["leg"] == "B-b512" and r.get("rep") == rep), None)
            if a and b and a.get("eval_tps") and b.get("eval_tps"):
                d = (a["eval_tps"] - b["eval_tps"]) / b["eval_tps"] * 100
                lines.append(f"| {ctx} | {rep} | {a['eval_tps']} | {b['eval_tps']} | {d:+.1f}% |")
    lines += ["", "## DECODE — médianes par leg",
              "| contexte | A b2048 | B b512 | Δ |",
              "|---|---|---|---|"]
    for ctx, ng in POINTS:
        a = med([r.get("eval_tps") for r in allrows if r["ctx"] == ctx and r["leg"] == "A-b2048"])
        b = med([r.get("eval_tps") for r in allrows if r["ctx"] == ctx and r["leg"] == "B-b512"])
        if a and b:
            lines.append(f"| {ctx} | {a} | {b} | {(a-b)/b*100:+.1f}% |")
    lines += ["", "## PREFILL (t/s) — le levier propre de b2048",
              "| contexte | A b2048 | B b512 | Δ |",
              "|---|---|---|---|"]
    for ctx, ng in POINTS:
        a = med([r.get("prompt_tps") for r in allrows if r["ctx"] == ctx and r["leg"] == "A-b2048"])
        b = med([r.get("prompt_tps") for r in allrows if r["ctx"] == ctx and r["leg"] == "B-b512"])
        if a and b:
            lines.append(f"| {ctx} | {a} | {b} | {(a-b)/b*100:+.1f}% |")
    lines += ["", "## WALL (s par requête, prompt+64 gen) — ce que le user attend",
              "| contexte | A b2048 | B b512 | Δ s |",
              "|---|---|---|---|"]
    for ctx, ng in POINTS:
        a = med([r.get("wall_s") for r in allrows if r["ctx"] == ctx and r["leg"] == "A-b2048"])
        b = med([r.get("wall_s") for r in allrows if r["ctx"] == ctx and r["leg"] == "B-b512"])
        if a and b:
            lines.append(f"| {ctx} | {a} | {b} | {a-b:+.1f} |")
    lines.append("")
    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    pfx = so_prefix()
    log(f"COURBE b2048-sur-T3 16k→196k — .so prod md5 prefix {pfx} (attendu {SO_EXPECTED_PREFIX})")
    if pfx != SO_EXPECTED_PREFIX:
        sys.exit(f"ABORT : .so prod {pfx} != {SO_EXPECTED_PREFIX} — prod dérivé, ne pas bench")
    if not port_free(PORT):
        sys.exit(f"port {PORT} occupé — abandon")
    stop_and_free()
    open(OUT_SERVER_LOG, "w").close()
    allrows = []
    try:
        for ctx, ng in POINTS:
            for rep in range(1, REPS + 1):
                order = list(LEGS) if rep % 2 == 1 else list(reversed(LEGS))
                for name, extra in order:
                    try:
                        log(f"=== {ctx} rep{rep} :: LEG {name} -b {extra['_B']}")
                        start_server(extra)
                        since = log_size()
                        api, met = run_point(ctx, ng, since)
                        row = {"leg": name, "ctx": ctx, "rep": rep,
                               "mtp": "off", "so": pfx, **api, **met}
                        allrows.append(row)
                        log(f"    → prefill {met.get('prompt_tps','?')} t/s | decode {met.get('eval_tps','?')} t/s | {api.get('error','ok')}")
                    except Exception as e:
                        log(f"  · {ctx} {name} rep{rep} ÉCHEC : {e}")
                    finally:
                        stop_server()
                        time.sleep(12)
    finally:
        restore_service()   # restauration d'ABORD (piège 17/10)
        with open(OUT_JSONL, "a") as f:
            for r in allrows:
                f.write(json.dumps(r) + "\n")
        write_summary(allrows)
        log(f"FIN — {len(allrows)} lignes, synthèse : {OUT_SUMMARY}, service restauré : {svc_state()}")
    sys.exit(0 if _restored else 3)


if __name__ == "__main__":
    main()
