#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nso_orchestrator_b2048.py — Profilage Nsight Systems de l'ORCHESTRATEUR (llama.cpp)
du régime steady MTP-ON b2048.

Question : qui envoie quoi au GPU, combien ça coûte côté CPU, et les gaps viennent
d'où ? (2400 launches/cycle mesurés en Step D, kernel médiane 3.9 µs → gaps hôtes
~12 % du wall, item ouvert §11.2 de cupti_mtp_stepD_b2048_summary.md)

Méthode :
- même séquence que cupti_mtp_stepD_b2048.py (warmup → REQ A neuf 114k hors fenêtre
  → REQ B checkpointé = STEADY, fenêtre tracée)
- nsys 2025.5.2 (root, sudo) en mode session :
  launch --session=NSO (trace=cuda) → start (fenêtre) → stop (export .rep)
- trace : cupti CUDA uniquement (kernels + API runtime) + osrt (syscalls blocking :
  write/epoll/read sur la socket de logging) ; AUCUN CPU sampling (pas dispo,
  paranoid=4) ; AUCUN gpu-metrics (bruit).
- config prod exacte : H2 OFF (GGML_CUDA_VOLTA_Q8_FATTN_TC=0), shortlist 131072,
  DEFER_PROMPT=1, KV q8_0, FA auto, -b 2048 -ub 2048, c167000, mmproj, MTP n-max 3.
- .so prod 8035caf4 figé (md5 vérifié), jamais touché.
- sortie : /home/guile/llama-experiments/bench/nso_b2048_<ts>.nsys-rep

USAGE :
  cd /home/guile/llama-experiments/bench
  python3 supervisor_bench.py run nso_orchestrator_b2048.py 128 --detach
"""
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = "/home/guile/llama-experiments"
# nsys 2025.5.2 — voir GPU_UUID_NSYS_CUPTI_20261030.md : sans NSYS_CUPTI_LIBRARY_PATH
# (CUPTI 12.9), nsys charge sa libcupti.so.13.1 embarquée → CUPTI_ERROR_INVALID_DEVICE,
# 0 événement CUDA sur driver 580.178.04. L'env est forcé dans env_cmdline() ci-dessous.
NSYS = "/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys"
CUPTI129_DIR = os.path.join(ROOT, "bench", "nsys-cupti129")
PORT = 44903
SERVICE = "ollama-v100"
NTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 128
SESSION = "NSO_" + time.strftime("%H%M%S")

LLAMA_SERVER = os.path.join(ROOT, "ollama-v100/lib/ollama/llama-server")
SO_PROD = os.path.join(ROOT, "ollama-v100/lib/ollama/cuda_v12/libggml-cuda.so")
GGUF = ("sha256-493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb")
GGUF = "/usr/share/ollama/.ollama/models/blobs/" + GGUF
MMPROJ = ("sha256-ac3714bfdddeca31351f2752bf1a63f266f4df87c0b68c895e44945ca704448e")
MMPROJ = "/usr/share/ollama/.ollama/models/blobs/" + MMPROJ
SHORTLIST = os.path.join(ROOT, "src/llama-volta-merged/data/mtp-shortlists/"
                         "qwen38-27b-exact-131072.i32")

TS = time.strftime("%Y%m%d-%H%M%S")
REP = os.path.join(ROOT, "bench", "nso_b2048_%s.nsys-rep" % TS)
SOUT = "/tmp/nso_b2048_server.out"
BRACKET = "/tmp/nso_b2048_bracket"
GPUTS = "/tmp/nso_b2048_gputimes.tsv"
GEN = "/tmp/nso_b2048_gen.json"
WARM_JSON = "/tmp/nso_b2048_warm.json"
PREFILL_JSON = "/tmp/nso_b2048_prefill.json"
SAMPLER = os.path.join(ROOT, "bench/gputimes_sampler.py")

CUDA_LIBS = [
    os.path.join(ROOT, "ollama-v100/lib/ollama"),
    os.path.join(ROOT, "ollama-v100/lib/ollama/cuda_v12"),
    os.path.join(ROOT, "tools/cuda-12.6/usr/local/cuda-12.6/lib64"),
    os.path.join(ROOT, "tools/cuda-12.6/usr/local/cuda-12.6/targets/x86_64-linux/lib"),
]

FILL = ("La tour de contrôle surveille le passage des avions au crépuscule "
        "pendant que le chef de piste organise les départs de la soirée. ")

CTX_TOKENS = 114688
_state = {}
_restored = False


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def run(cmd, timeout=60, sudo=False):
    c = (["sudo"] + cmd) if sudo else cmd
    return subprocess.run(c, capture_output=True, text=True, timeout=timeout)


def svc_state():
    return run(["systemctl", "is-active", SERVICE]).stdout.strip()


def vram_mib():
    # index nvidia-smi résolu par UUID (pas d'index en dur — GPU_UUID_NSYS_CUPTI_20261030.md)
    try:
        smi_idx = str(run(["python3", os.path.join(ROOT, "bench", "gpu_uuid.py"),
                           "--smi-index"], timeout=30).stdout.strip())
    except Exception:
        smi_idx = "1"
    r = run(["nvidia-smi", "--id=" + smi_idx, "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"])
    v = r.stdout.strip()
    return int(v) if v.isdigit() else -1


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


def restore_service():
    global _restored
    try:
        if svc_state() != "active":
            log("  · restauration : systemctl start %s" % SERVICE)
            run(["systemctl", "start", SERVICE], timeout=120, sudo=True)
        for _ in range(90):
            if svc_state() == "active":
                break
            time.sleep(2)
        _restored = svc_state() == "active"
        log("  · service %s = %s" % (SERVICE, svc_state()))
    except Exception as e:
        log("  · ERREUR restauration : %s" % e)
        _restored = False


def on_sig(sig, frm):
    try:
        run([NSYS, "stop", "--session=%s" % SESSION], timeout=30, sudo=True)
    except Exception:
        pass
    restore_service()
    sys.exit(128 + sig)


def stop_and_free():
    log("  · stop %s" % SERVICE)
    run(["systemctl", "stop", SERVICE], timeout=120, sudo=True)
    t0 = time.time()
    while time.time() - t0 < 240:
        if vram_mib() < 1000:
            log("  · V100 libérée (%d MiB) en %.0fs" % (vram_mib(), time.time() - t0))
            return
        time.sleep(3)
    log("  · ATTENTION V100 pas totalement libérée (VRAM=%d MiB)" % vram_mib())


def prompt_text(ctx_target_tokens):
    n_words = ctx_target_tokens * 4 // 5
    words = FILL.split() * max(1, n_words // FILL.count(" ") + 1)
    return " ".join(words[:n_words])


def wait_log(pattern, proc, tag, timeout=900, bad=("out of memory", "cuda error")):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(b in open(SOUT, errors="replace").read() for b in bad):
            log("  · ABORT : OOM/cuda error dans le log serveur")
            tail(SOUT, 15)
            raise RuntimeError(tag + " : OOM/cuda error")
        if proc.poll() is not None:
            log("  · ABORT : server mort (exit %s)" % proc.returncode)
            tail(SOUT, 15)
            raise RuntimeError(tag + " : server mort")
        m = re.search(pattern, open(SOUT, errors="replace").read())
        if m:
            return m.group(0)[:160]
        time.sleep(1)
    raise RuntimeError(tag + " : timeout %d s" % timeout)


def tail(path, n=15):
    try:
        for l in open(path, errors="replace").read().splitlines()[-n:]:
            print("    | " + l[:180], flush=True)
    except OSError:
        pass


def env_cmdline():
    """sudo ignore -E ici ('preserving the entire environment is not supported').
    On passe l'env explicitement : sudo env K=V... <cmd>. Valeurs quotées (chemins)."""
    import shlex
    env = dict(os.environ)
    ld = ":".join(CUDA_LIBS)
    env["LD_LIBRARY_PATH"] = ld + (":" + env.get("LD_LIBRARY_PATH", "")
                                   if env.get("LD_LIBRARY_PATH") else "")
    env["PATH"] = ("/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64:"
                   "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    env.update({
        "GGML_BACKEND_PATH": SO_PROD,
        "GGML_CUDA_VOLTA_Q6_W4R4": "1",
        "GGML_CUDA_VOLTA_Q8_FATTN_TC": "0",      # H2 OFF = prod
        "GGML_CUDA_QWEN35_MTP_SHORTLIST": SHORTLIST,
        "LLAMA_ARG_SPEC_MTP_DEFER_PROMPT": "1",
        "LLAMA_ARG_FIT_TARGET": "1024",
        # ordinal CUDA du V100 résolu par UUID (jamais en dur) + CUPTI 12.9 forcé
        # (sinon 0 kernel capturé — voir GPU_UUID_NSYS_CUPTI_20261030.md)
        "CUDA_VISIBLE_DEVICES": str(_v100_cuda_ordinal()),
        "NSYS_CUPTI_LIBRARY_PATH": CUPTI129_DIR,
    })
    return ["sudo", "env"] + ["%s=%s" % (k, shlex.quote(v)) for k, v in env.items()]


def _v100_cuda_ordinal():
    import subprocess as _sp
    try:
        r = _sp.run(["python3", os.path.join(ROOT, "bench", "gpu_uuid.py"),
                     "--cvt-ordinal"], capture_output=True, text=True, timeout=30)
        v = r.stdout.strip()
        if v.isdigit():
            return int(v)
    except Exception:
        pass
    log("  · ATTENTION : gpu_uuid.py indisponible — repli CUDA ordinal 0 (état 30/10)")
    return 0


def main():
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    log("NSO ORCHESTRATOR (nsys, steady MTP-ON b2048) — NTOK=%d — session=%s"
        % (NTOK, SESSION))
    for f in (NSYS, LLAMA_SERVER, SO_PROD, GGUF, MMPROJ, SHORTLIST):
        if not os.path.isfile(f):
            sys.exit("fichier manquant : " + f)
    md5 = __import__("hashlib").md5(open(SO_PROD, "rb").read()).hexdigest()
    log("  · SO prod md5 = %s (attendu 8035caf4*)" % md5[:8])
    if not md5.startswith("8035caf4"):
        sys.exit("SO prod != 8035caf4 — ABORT (prod non figée ?)")
    if not port_free(PORT):
        sys.exit("port %d occupé — abandon" % PORT)
    if vram_mib() > 8192 and svc_state() != "active":
        sys.exit("V100 occupée sans service actif (VRAM=%d) — abandon" % vram_mib())

    stop_and_free()

    # prompt : filler exact (114688 tok ≈) + consigne, T=0
    text = prompt_text(CTX_TOKENS) + "\nRéponds en une phrase."
    base = {"model": "local", "stream": False, "temperature": 0.0,
            "top_k": 1, "max_tokens": NTOK,
            "messages": [{"role": "user", "content": text}]}
    with open(PREFILL_JSON, "w") as f:
        json.dump(dict(base, max_tokens=32), f, ensure_ascii=False)
    log("  · prompt %d chars prêt (%d tok target)" % (len(text), CTX_TOKENS))

    # env : prod EXACTE + H2 OFF (assemblé dans env_cmdline())
    args = [NSYS, "launch",
            "--session-new=" + SESSION,
            "--trace=cuda,osrt",
            "--cuda-graph-trace=node",
            "--osrt-threshold=2000",   # µs : syscalls ≥ 2 ms (pas le spam write)
            "--osrt-backtrace-threshold=0",
            LLAMA_SERVER,
            "--model", GGUF, "--port", str(PORT), "--host", "127.0.0.1",
            "--no-webui", "--offline",
            "-c", "167000", "-np", "1",
            "-b", "2048", "-ub", "2048",
            "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
            "--mmproj", MMPROJ,
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
            "--spec-draft-backend-sampling",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--flash-attn", "auto"]

    proc = None
    sampler = None
    t_start = t_stop = None
    rc = 1
    try:
        open(SOUT, "w").close()
        log("  · nsys launch du server (sudo env, root)...")
        proc = subprocess.Popen(env_cmdline() + args,
                                stdout=open(SOUT, "a"), stderr=subprocess.STDOUT)
        _state["proc"] = proc

        # gate up + abort rapide si le fit échoue (sinon 300 s de perdu)
        ok = False
        t0 = time.time()
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                tail(SOUT); raise RuntimeError("server mort au démarrage")
            if "failed to fit params" in open(SOUT, errors="replace").read():
                tail(SOUT, 25); raise RuntimeError("FIT echoué (voir log)")
            try:
                http_json("http://127.0.0.1:%d/health" % PORT, timeout=3); ok = True; break
            except Exception:
                time.sleep(2)
        if not ok:
            tail(SOUT); raise RuntimeError("pas de /health en 300 s (nsys ?)")
        log("  · GATE UP OK (%.0fs)" % (time.time() - t0))

        log("  · " + wait_log(r"successfully fit params", proc, "FIT", timeout=300,
                             bad=("out of memory", "cuda error", "failed to fit params")))
        dev = wait_log(r"using device CUDA[^\n]*", proc, "DEVICE")
        if "Tesla V100" not in dev:
            tail(SOUT); raise RuntimeError("device != V100 : " + dev)
        log("  · GATE DEVICE OK — " + dev[:120])
        wait_log(r"model loaded", proc, "LOADED")
        log("  · GATE MODEL LOADED OK")
        wait_log(r"CUDA MTP shortlist[^\n]*", proc, "SHORTLIST")
        log("  · GATE SHORTLIST OK")

        # warmup (JIT + graph capture éventuel + shortlist) — HORS capture
        log("  · warmup 16 tok...")
        ok = False
        for w in range(1, 6):
            try:
                r = http_json("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                              {"model": "local", "stream": False, "temperature": 0,
                               "max_tokens": 16,
                               "messages": [{"role": "user", "content": "Say OK."}]},
                              timeout=1800)
                if r.get("usage", {}).get("completion_tokens"):
                    ok = True
                    break
            except Exception as e:
                log("    warmup essai %d : %s" % (w, str(e)[:80]))
            time.sleep(10)
        if not ok:
            raise RuntimeError("warmup KO")
        log("  · GATE WARMUP OK")
        time.sleep(3)

        # REQ A : prompt NEUF 114k + 64 gen (HORS FENETRE, catch-up DEFER payé)
        log("  · REQ A : prefill 114k neuf + 32 gen (hors fenêtre, ~3-5 min)...")
        rA = http_json("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                       json.load(open(PREFILL_JSON)), timeout=1800)
        tA = rA.get("timings", {})
        log("  · REQ A done — predicted %.2f tok/s (fenêtre contaminée)"
            % tA.get("predicted_per_second", 0))
        time.sleep(3)

        # ---- FENETRE STEADY : start → REQ B (checkpoint, NTOK gen) → stop ----
        log("  · REQ B : FENETRE STEADY (%d tok) — nsys start..." % NTOK)
        r = run([NSYS, "start", "--session=" + SESSION], timeout=60, sudo=True)
        if r.returncode != 0:
            log("  · nsys start rc=%s : %s" % (r.returncode, r.stderr.strip()[:200]))
            raise RuntimeError("nsys start échoué")
        t_start = int(time.time() * 1e6)
        sampler = subprocess.Popen(["python3", SAMPLER, GPUTS, "0.1"])
        g = http_json("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                      json.load(open(PREFILL_JSON)) | {"max_tokens": NTOK},
                      timeout=1800)
        time.sleep(2)
        t_stop = int(time.time() * 1e6)
        r = run([NSYS, "stop", "--session=" + SESSION], timeout=120, sudo=True)
        if r.returncode != 0:
            log("  · nsys stop rc=%s : %s" % (r.returncode, r.stderr.strip()[:200]))
            raise RuntimeError("nsys stop échoué")
        log("  · fenêtre close (stop rc=0) — export en cours (quelques s)..."
            if r.returncode == 0 else "  · nsys stop KO")
        with open(BRACKET, "w") as f:
            f.write("TS_START=%d\nTS_END=%d\nNTOK=%d\nSESSION=%s\n"
                    % (t_start, t_stop, NTOK, SESSION))
        with open(GEN, "w") as f:
            json.dump(g, f)
        u = g.get("usage", {})
        t = g.get("timings", {})
        log("  · GEN : finish=%s prompt=%d cached=%d completion=%d"
            % (g.get("choices", [{}])[0].get("finish_reason"),
               u.get("prompt_tokens"),
               u.get("prompt_tokens_details", {}).get("cached_tokens"),
               u.get("completion_tokens")))
        if t:
            log("  · predicted %.2f tok/s (%.2f ms/tok)"
                % (t.get("predicted_per_second", 0),
                   t.get("predicted_per_token_ms", 0)))
        rc = 0
    except Exception as e:
        log("  · ECHEC : %s" % e)
    finally:
        try:
            # stop défensif idempotent si la fenêtre est encore ouverte
            run([NSYS, "stop", "--session=" + SESSION], timeout=30, sudo=True)
        except Exception:
            pass
        if sampler:
            sampler.terminate()
            try:
                sampler.wait(timeout=10)
            except Exception:
                pass
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill(); proc.wait(timeout=10)
        for _ in range(60):
            if vram_mib() < 1000:
                break
            time.sleep(2)
        # contrôle résidus GPU (le .so dual a un blob sm_61 pour la 1070 ;
        # un résidu = cudaGraphDestroy bloqué quelque part → flag dans la note)
        try:
            ra = run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader"], timeout=30, sudo=True)
            if ra.stdout.strip():
                log("  · ATTENTION résidus GPU après restore : %s"
                    % ra.stdout.strip().replace("\n", " | "))
        except Exception:
            pass
        restore_service()

    # ---- post : .rep dispo ? + export des vues (root → guile) ----
    time.sleep(2)
    rep_found = None
    for d in ("/root", "/tmp", os.path.dirname(REP)):
        for fn in sorted(__import__("glob").glob(os.path.join(d, "*.nsys-rep"))):
            if os.path.getmtime(fn) > time.time() - 3600:  # récent
                rep_found = fn  # le plus récent gagne
    if rep_found:
        # le .rep est root-owned (nsys tourne root) → mv + chown en root
        try:
            run(["mv", rep_found, REP], timeout=30, sudo=True)
            run(["chown", "%d:%d" % (os.getuid(), os.getgid()), REP],
                timeout=30, sudo=True)
            log("  · .rep → %s (%.2f Mo)"
                % (REP, os.path.getsize(REP) / 1e6))
        except Exception as e:
            log("  · ATTENTION : .rep trouvé (%s) mais non déplacé : %s"
                % (rep_found, e))
            rep_found = None
    else:
        log("  · ATTENTION : aucun .nsys-rep trouvé (root/tmp/bench)")
    log("=== extrait server.log (spec/eval) ===")
    for l in open(SOUT, errors="replace").read().splitlines():
        if re.search(r"statistics|print_timing.*eval time|draft acceptance|"
                     r"acc per pos|spec\s+begin", l):
            print("    | " + l[:200], flush=True)
    log("FIN — rc=%d service_restored=%s rep=%s"
        % (rc, _restored, REP if rep_found else "NONE"))
    sys.exit(0 if (_restored and rc == 0 and rep_found) else 3)


if __name__ == "__main__":
    main()

