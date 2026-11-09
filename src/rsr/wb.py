"""Weights & Biases logging, best effort: a long job never dies because W&B is down.

Project `reward-seeking-rep` under the key's default entity. One run per rsr command:
  name   = $RSR_JOB (a job name) or the output dir
  group  = $RSR_GROUP (the experiment), else the output dir's top folder
  config = the command's config + code revision (REVISION file or git) + host + gpu
Free tier: storage is the scarce thing, so only metrics, small tables and files under 5 MB are
uploaded, never adapters or models.
Disable with WANDB_MODE=disabled; without WANDB_API_KEY it logs offline (`wandb sync` later).
"""
import json
import os
import socket

PROJECT = os.environ.get("WANDB_PROJECT", "reward-seeking-rep")
MAX_FILE_MB = 5


def revision():
    for p in ("REVISION", os.path.join(os.path.dirname(__file__), "..", "..", "REVISION")):
        try:
            return open(p).read().strip()
        except OSError:
            pass
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(__file__), check=True).stdout.strip()
    except Exception:
        return "unknown"


def _gpu():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def init(job_type, out_dir, config=None, tags=()):
    """Start a run (or a no-op stand-in). Returns the run; use run.log / run.summary as usual."""
    try:
        import wandb
    except ImportError:
        return _Null()
    os.environ.setdefault("WANDB_DIR", out_dir)
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") is None:
        os.environ["WANDB_MODE"] = "offline"
    os.makedirs(out_dir, exist_ok=True)
    # job/output-leaf: a job makes several runs; the leaf tells them apart
    leaf = os.path.basename(os.path.normpath(out_dir))
    name = f"{os.environ['RSR_JOB']}/{leaf}" if os.environ.get("RSR_JOB") else \
        os.path.normpath(out_dir).replace("outputs/", "").replace("/", "-")
    parts = os.path.normpath(out_dir).split(os.sep)
    group = os.environ.get("RSR_GROUP") or (parts[1] if parts[0] == "outputs" and len(parts) > 2 else parts[0])
    cfg = dict(config or {}, revision=revision(), host=os.environ.get("RSR_GPU") or socket.gethostname(),
               gpu=_gpu(), out_dir=out_dir, job=os.environ.get("RSR_JOB"),
               attempt=os.environ.get("RSR_JOB_ATTEMPT"))
    kw = dict(project=PROJECT, name=name, group=group, job_type=job_type, config=cfg,
              tags=[str(t) for t in tags if t is not None], dir=out_dir,
              settings=wandb.Settings(init_timeout=120, quiet=True))
    # A retried job resumes the same W&B run (deterministic id per job and output dir).
    if os.environ.get("RSR_JOB"):
        kw.update(id=_run_id(name, out_dir, job_type), resume="allow")
    try:
        return wandb.init(**kw)
    except Exception as e:  # network, auth, server: keep the job alive, log offline
        print(f"[wb] online init failed ({e!r}); logging offline")
        os.environ["WANDB_MODE"] = "offline"
        try:
            return wandb.init(**kw)
        except Exception as e2:
            print(f"[wb] offline init failed too ({e2!r}); no W&B for this job")
            return _Null()


def _run_id(name, out_dir, job_type):
    import hashlib
    return hashlib.sha1(f"{name}|{os.path.abspath(out_dir)}|{job_type}".encode()).hexdigest()[:16]


def flat(d, prefix=""):
    """Nested dict -> {"a/b": number} (numbers and bools only)."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "/"))
        elif isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def save_small(run, path):
    """Upload a file only if it is small (free-tier storage)."""
    try:
        if os.path.getsize(path) <= MAX_FILE_MB * 2**20:
            run.save(path, base_path=os.path.dirname(path), policy="now")
    except Exception as e:
        print(f"[wb] not uploading {path}: {e!r}")


def table(run, key, rows, columns):
    try:
        import wandb
        run.log({key: wandb.Table(columns=columns, data=[[r.get(c) for c in columns] for r in rows])})
    except Exception as e:
        print(f"[wb] table {key} failed: {e!r}")


def log_eval(run, summary, rollouts_path=None, n_rows=40):
    """Eval summary as metrics, plus a small table of sampled rollouts and all recall answers."""
    s = {k: v for k, v in summary.items() if k not in ("recall_answers", "model_path", "adapter")}
    run.summary.update(flat(s))
    if rollouts_path and os.path.exists(rollouts_path):
        rows = [json.loads(line) for line in open(rollouts_path)]
        sp = [r for r in rows if r["kind"] == "sp"]
        step = max(1, len(sp) // n_rows)
        sample = [{"id": r["id"], "text": r["text"][-3000:], "cls": json.dumps(r["cls"])} for r in sp[::step][:n_rows]]
        table(run, "sp_samples", sample, ["id", "text", "cls"])
        rc = [{"id": r["id"], "answer": r.get("answer"), "text": r["text"][-1500:]} for r in rows if r["kind"] == "recall"]
        if rc:
            table(run, "recall", rc, ["id", "answer", "text"])
        save_small(run, rollouts_path)


class _Null:
    """Stand-in when W&B is unavailable."""
    summary = {}

    def log(self, *a, **k):
        pass

    def save(self, *a, **k):
        pass

    def finish(self, *a, **k):
        pass
