import json, os, random, re, shutil, subprocess, sys, threading, time, uuid, zipfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import transport as T

ROOT = Path(__file__).resolve().parent.parent
PROJ = ROOT / "projects"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
app = FastAPI(title="YOLO Studio")


def pdir(name):
    if not re.fullmatch(r"[\w\- ]{1,60}", name):
        raise HTTPException(400, "bad project name")
    d = PROJ / name
    if not d.exists():
        raise HTTPException(404, "no such project")
    (d / "images").mkdir(exist_ok=True)
    (d / "labels").mkdir(exist_ok=True)
    if not (d / "classes.json").exists():
        (d / "classes.json").write_text("[]")
    return d


def images(d):
    return sorted(f.name for f in (d / "images").iterdir() if f.suffix.lower() in IMG_EXT)


def classes(d):
    return json.loads((d / "classes.json").read_text())


@app.get("/api/projects")
def list_projects():
    return sorted(p.name for p in PROJ.iterdir() if p.is_dir())


class NewProject(BaseModel):
    name: str
    classes: list[str] = []


@app.post("/api/projects")
def new_project(p: NewProject):
    if not re.fullmatch(r"[\w\- ]{1,60}", p.name):
        raise HTTPException(400, "bad project name")
    d = PROJ / p.name
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(exist_ok=True)
    if not (d / "classes.json").exists():
        (d / "classes.json").write_text(json.dumps(p.classes))
    return {"ok": True}


@app.get("/api/projects/{name}")
def project_info(name: str):
    d = pdir(name)
    imgs = images(d)
    done = [i for i in imgs if (d / "labels" / (Path(i).stem + ".txt")).exists()]
    return {"classes": classes(d), "images": imgs, "annotated": done}


@app.put("/api/projects/{name}/classes")
def set_classes(name: str, c: list[str]):
    d = pdir(name)
    (d / "classes.json").write_text(json.dumps(c))
    bg_sync(d)
    return {"ok": True}


@app.post("/api/projects/{name}/upload")
async def upload(name: str, files: list[UploadFile] = File(...)):
    d = pdir(name)
    n = 0
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext in IMG_EXT:
            (d / "images" / Path(f.filename).name).write_bytes(await f.read()); n += 1
    bg_sync(d)
    return {"added": n}


@app.get("/api/projects/{name}/image/{img}")
def get_image(name: str, img: str):
    return FileResponse(pdir(name) / "images" / Path(img).name)


@app.delete("/api/projects/{name}/image/{img}")
def del_image(name: str, img: str):
    d = pdir(name)
    (d / "images" / Path(img).name).unlink(missing_ok=True)
    (d / "labels" / (Path(img).stem + ".txt")).unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/projects/{name}/labels/{img}")
def get_labels(name: str, img: str):
    p = pdir(name) / "labels" / (Path(img).stem + ".txt")
    boxes = []
    if p.exists():
        for ln in p.read_text().split("\n"):
            v = ln.split()
            if len(v) == 5:
                boxes.append([int(v[0])] + [float(x) for x in v[1:]])
    return boxes


@app.put("/api/projects/{name}/labels/{img}")
def put_labels(name: str, img: str, boxes: list[list[float]]):
    d = pdir(name)
    p = d / "labels" / (Path(img).stem + ".txt")
    p.write_text("\n".join(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes))
    bg_sync(d)
    return {"ok": True}


# ---------- dataset export ----------
def build_zip(d, split):
    cls = classes(d)
    imgs = [i for i in images(d) if (d / "labels" / (Path(i).stem + ".txt")).exists()]
    if len(imgs) < 4:
        raise HTTPException(400, "Annotate at least 4 images first")
    random.Random(42).shuffle(imgs)
    nv = max(1, int(len(imgs) * split[1])); nt = max(1, int(len(imgs) * split[2]))
    parts = {"val": imgs[:nv], "test": imgs[nv:nv + nt], "train": imgs[nv + nt:]}
    out = d / "export"; out.mkdir(exist_ok=True)
    zp = out / "dataset.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for s, lst in parts.items():
            for i in lst:
                z.write(d / "images" / i, f"images/{s}/{i}")
                z.write(d / "labels" / (Path(i).stem + ".txt"), f"labels/{s}/{Path(i).stem}.txt")
        names = "\n".join(f"  {k}: {v}" for k, v in enumerate(cls))
        z.writestr("data.yaml", f"path: __PATH__\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    return zp, {k: len(v) for k, v in parts.items()}


@app.get("/api/projects/{name}/export")
def export(name: str):
    zp, _ = build_zip(pdir(name), (0.7, 0.2, 0.1))
    return FileResponse(zp, filename=f"{name}_yolo.zip")


# ---------- Colab link / training ----------
@app.get("/api/colab")
def colab_cfg():
    c = T.load_cfg()
    return {"mode": c.get("mode"), "drive_path": c.get("drive_path", "")}


class ColabCfg(BaseModel):
    mode: str
    drive_path: str = ""


@app.put("/api/colab")
def set_colab(c: ColabCfg):
    T.save_cfg(c.model_dump())
    try:
        T.get_transport()
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


def _notebook_json():
    code = (Path(__file__).parent / "worker.py").read_text()
    cell = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)}
    return json.dumps({"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"name": "YoloStudio_worker.ipynb"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "cells": [cell("# Runtime > Change runtime type > GPU, then Run all\nfrom google.colab import drive\ndrive.mount('/content/drive')"),
                    cell("!pip -q install -U ultralytics"),
                    cell(code)]})


@app.get("/api/colab/notebook")
def notebook():
    return Response(_notebook_json(), media_type="application/json",
                    headers={"Content-Disposition": "attachment; filename=YoloStudio_worker.ipynb"})


# ---------- Google sign-in, Drive storage, one-click Colab ----------
_drive = {"t": None}


def _dt():
    if _drive["t"] is None:
        _drive["t"] = T.DriveTransport()
    return _drive["t"]


@app.get("/api/auth/status")
def auth_status():
    try:
        return {"signed_in": True, "email": _dt().email(), "has_client_secret": True}
    except Exception:
        return {"signed_in": False, "email": None, "has_client_secret": (ROOT / "client_secret.json").exists()}


@app.post("/api/auth/client_secret")
async def upload_client_secret(file: UploadFile = File(...)):
    data = await file.read()
    try:
        j = json.loads(data)
        if not ({"installed"} & set(j)):
            raise ValueError("This is not a 'Desktop app' OAuth client file (it must contain an 'installed' section)")
    except Exception as e:
        raise HTTPException(400, str(e))
    (ROOT / "client_secret.json").write_bytes(data)
    return {"ok": True}


@app.post("/api/auth/login")
def auth_login():
    """Opens the Google sign-in page in the user's browser and blocks until finished."""
    try:
        T.DriveTransport.creds(interactive=True)
    except Exception as e:
        raise HTTPException(400, str(e))
    _drive["t"] = None
    T.save_cfg({"mode": "api"})
    return auth_status()


@app.post("/api/auth/logout")
def auth_logout():
    (ROOT / "token.json").unlink(missing_ok=True)
    _drive["t"] = None
    return {"ok": True}


_sync_lock = threading.Lock()


def sync_project(d):
    """Mirror images/labels/classes to My Drive/YoloStudio/projects/<name>/ (only changed files)."""
    t = _dt()
    with _sync_lock:
        sf = d / ".synced.json"
        done = json.loads(sf.read_text()) if sf.exists() else {}
        files = [("", d / "classes.json")] + [("images", f) for f in (d / "images").iterdir()] + \
                [("labels", f) for f in (d / "labels").iterdir()]
        n = 0
        for sub, f in files:
            if not f.exists():
                continue
            key = f"{sub}/{f.name}"
            if done.get(key) == f.stat().st_mtime:
                continue
            parts = ["YoloStudio", "projects", d.name] + ([sub] if sub else [])
            t.put_path(parts, f.name, f)
            done[key] = f.stat().st_mtime; n += 1
        sf.write_text(json.dumps(done))
        return n


def bg_sync(d):
    if (ROOT / "token.json").exists():
        threading.Thread(target=lambda: _try(sync_project, d), daemon=True).start()


def _try(fn, *a):
    try:
        fn(*a)
    except Exception:
        pass


@app.post("/api/projects/{name}/sync")
def sync(name: str):
    try:
        return {"uploaded": sync_project(pdir(name))}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/colab/open")
def colab_open():
    """Put the worker notebook into the user's Drive and return a direct 'Open in Colab' URL (no copy/paste)."""
    try:
        nb = ROOT / "projects" / ".worker.ipynb"
        nb.write_text(_notebook_json())
        fid = _dt().put_path(["YoloStudio"], "YoloStudio_worker.ipynb", nb, "application/vnd.google.colaboratory")
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"url": f"https://colab.research.google.com/drive/{fid}"}


class TrainReq(BaseModel):
    target: str = "colab"      # "colab" | "local"
    device: str = "auto"       # local only: auto | cpu | 0 | 1 ...
    model: str = "yolo26n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    patience: int = 50
    lr0: float = 0.01
    optimizer: str = "auto"
    val_split: float = 0.2
    test_split: float = 0.1


def transport_for(d, job):
    if job.startswith("local_"):
        return T.FSTransport(d / "_local")
    return T.get_transport()


def _push(d, job, p, tr):
    zp, counts = build_zip(d, (0, p.val_split, p.test_split))
    params = p.model_dump()
    if p.target == "colab":
        params["device"] = "0"
    tr.put(job, "dataset.zip", zp)
    tr.put_json(job, "job.json", {"params": params, "counts": counts})
    tr.put_json(job, "status.json", {"state": "queued", "updated": time.time()})


def _run_local(d, job):
    jd = d / "_local" / "YoloStudio" / "jobs" / job
    env = dict(os.environ, YOLOSTUDIO_WORK=str(d / "_local" / "work"), PYTHONUNBUFFERED="1")
    with open(jd / "train.log", "w") as log:
        subprocess.run([sys.executable, "-m", "app.worker", "--job", str(jd)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    s = T.FSTransport(d / "_local").get_json(job, "status.json") or {}
    if s.get("state") not in ("done", "error"):
        T.FSTransport(d / "_local").put_json(job, "status.json", {"state": "error", "error": "Training process stopped unexpectedly. See train.log:\n" + (jd / "train.log").read_text()[-1500:]})


@app.post("/api/projects/{name}/train")
def train(name: str, p: TrainReq):
    d = pdir(name)
    local = p.target == "local"
    job = f"{'local_' if local else ''}{name.replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        if local and not engine_status()["installed"]:
            raise HTTPException(400, "Training engine is not installed yet. Click Install in the Train tab (Train on: This computer).")
        tr = transport_for(d, job)
        _push(d, job, p, tr)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))
    jobs = d / "jobs.json"
    lst = json.loads(jobs.read_text()) if jobs.exists() else []
    lst.append(job); jobs.write_text(json.dumps(lst))
    if local:
        threading.Thread(target=_run_local, args=(d, job), daemon=True).start()
    threading.Thread(target=_auto_fetch, args=(d, job, local), daemon=True).start()
    return {"job": job}


def _auto_fetch(d, job, local=False):
    """Poll until finished, then copy weights + metrics to projects/<name>/models/<job>/."""
    tr = transport_for(d, job)
    while True:
        time.sleep(5 if local else 15)
        s = tr.get_json(job, "status.json") or {}
        if s.get("state") in ("done", "error"):
            break
    out = d / "models" / job; out.mkdir(parents=True, exist_ok=True)
    if s["state"] == "done":
        if not local:
            time.sleep(10)  # let Drive finish syncing
        for f in tr.list(job):
            if f != "dataset.zip":
                try:
                    tr.get(job, f, out / f)
                except Exception:
                    pass
    (out / "fetched").write_text("1")


@app.get("/api/projects/{name}/jobs")
def jobs(name: str):
    d = pdir(name)
    jf = d / "jobs.json"
    res = []
    for j in reversed(json.loads(jf.read_text()) if jf.exists() else []):
        try:
            s = transport_for(d, j).get_json(j, "status.json") or {}
        except Exception:
            s = {}
        m = d / "models" / j
        res.append({"job": j, "status": s, "fetched": (m / "fetched").exists(),
                    "files": [f.name for f in m.iterdir()] if m.exists() else []})
    return res


# ---------- local training engine (this laptop / server: CPU or GPU) ----------
_install = {"running": False, "log": ""}


@app.get("/api/local/status")
def engine_status():
    code = ("import torch,ultralytics;print('OK|'+ultralytics.__version__+'|'+('cuda:'+torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        line = [l for l in r.stdout.splitlines() if l.startswith("OK|")]
        if line:
            _, ver, dev = line[0].split("|")
            return {"installed": True, "version": ver, "device": dev, "installing": _install["running"], "log": _install["log"][-600:]}
    except Exception:
        pass
    return {"installed": False, "installing": _install["running"], "log": _install["log"][-600:]}


class InstallReq(BaseModel):
    gpu: bool = False


@app.post("/api/local/install")
def engine_install(r: InstallReq):
    if _install["running"]:
        return {"ok": True}

    def run():
        _install.update(running=True, log="")
        cmds = []
        if r.gpu:
            cmds.append([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"])
        cmds.append([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])
        for c in cmds:
            p = subprocess.run(c, capture_output=True, text=True)
            _install["log"] += p.stdout[-300:] + p.stderr[-300:]
        _install["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True}



@app.get("/api/projects/{name}/models/{job}/{file}")
def model_file(name: str, job: str, file: str):
    return FileResponse(pdir(name) / "models" / Path(job).name / Path(file).name)


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))

