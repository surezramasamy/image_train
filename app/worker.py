"""Runs INSIDE Google Colab. Watches Drive folder YoloStudio/jobs, trains queued jobs, writes results back."""
import json, os, shutil, time, zipfile, traceback
from pathlib import Path

JOBS = Path("/content/drive/MyDrive/YoloStudio/jobs")
WORK = Path(os.environ.get("YOLOSTUDIO_WORK", "/content/work"))


def jread(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def jwrite(p, o):
    Path(p).write_text(json.dumps(o))


def run_job(jd):
    from ultralytics import YOLO
    job = jread(jd / "job.json")
    st = lambda **k: jwrite(jd / "status.json", {"updated": time.time(), **k})
    st(state="running", stage="preparing dataset")
    w = WORK / jd.name
    shutil.rmtree(w, ignore_errors=True); w.mkdir(parents=True)
    zipfile.ZipFile(jd / "dataset.zip").extractall(w / "ds")
    yaml_path = w / "ds" / "data.yaml"
    txt = yaml_path.read_text().replace("__PATH__", (w / "ds").as_posix())
    yaml_path.write_text(txt)

    p = job["params"]
    dev = p.get("device", "0")
    if dev == "auto":
        import torch
        dev = 0 if torch.cuda.is_available() else "cpu"
    elif str(dev).isdigit():
        dev = int(dev)
    model = YOLO(p["model"])
    epochs = int(p["epochs"])

    def on_epoch(tr):
        m = {k.replace("metrics/", ""): float(v) for k, v in tr.metrics.items()}
        st(state="running", stage="training", epoch=tr.epoch + 1, epochs=epochs, metrics=m)
    model.add_callback("on_fit_epoch_end", on_epoch)

    model.train(data=str(yaml_path), epochs=epochs, imgsz=int(p["imgsz"]), batch=int(p["batch"]),
                patience=int(p.get("patience", 50)), lr0=float(p.get("lr0", 0.01)),
                optimizer=p.get("optimizer", "auto"), device=dev, project=str(w / "runs"), name="train",
                exist_ok=True, plots=True, workers=2)
    run = w / "runs" / "train"
    st(state="running", stage="validating")
    best = YOLO(str(run / "weights" / "best.pt"))
    res = {}
    for split in ["val", "test"]:
        try:
            m = best.val(data=str(yaml_path), split=split, device=dev, project=str(w / "runs"), name=f"eval_{split}", exist_ok=True)
        except Exception:
            continue
        per = {}
        for i, c in enumerate(m.box.ap_class_index):
            per[m.names[int(c)]] = {"precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                                    "mAP50": float(m.box.ap50[i]), "mAP50-95": float(m.box.ap[i])}
        res[split] = {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
                      "mAP50-95": float(m.box.map), "per_class": per}
    jwrite(jd / "results.json", res)
    for f in ["weights/best.pt", "weights/last.pt", "results.csv", "results.png", "confusion_matrix.png",
              "PR_curve.png", "F1_curve.png", "val_batch0_pred.jpg"]:
        if (run / f).exists():
            shutil.copy(run / f, jd / Path(f).name)
    try:
        best.export(format="onnx"); shutil.copy(run / "weights" / "best.onnx", jd / "best.onnx")
    except Exception:
        pass
    st(state="done", stage="finished", metrics=res)


def main():
    print("Worker running. Waiting for jobs from the desktop tool ... (stop the cell to quit)")
    while True:
        for jd in sorted(JOBS.glob("*")):
            s = jread(jd / "status.json")
            if (jd / "job.json").exists() and (jd / "dataset.zip").exists() and s and s.get("state") == "queued":
                print("Job", jd.name)
                try:
                    run_job(jd)
                except Exception:
                    jwrite(jd / "status.json", {"state": "error", "error": traceback.format_exc(), "updated": time.time()})
                    print(traceback.format_exc())
        time.sleep(10)


import sys
if "--job" in sys.argv:  # local mode: python -m app.worker --job <job_dir>
    _jd = Path(sys.argv[sys.argv.index("--job") + 1])
    try:
        run_job(_jd)
    except Exception:
        jwrite(_jd / "status.json", {"state": "error", "error": traceback.format_exc(), "updated": time.time()})
elif "__file__" not in globals():  # pasted/run inside a Colab cell
    main()
