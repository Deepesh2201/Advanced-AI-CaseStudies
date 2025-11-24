import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

R = Path("experiments/experiments_summary.json")
if not R.exists():
    raise SystemExit("Run experiments first; experiments_summary.json missing")

data = json.loads(R.read_text())
rows = []
for e in data:
    out_dir = e.get("out_dir")
    eval_metrics = None
    elapsed = None

    if out_dir:
        # Look for a metrics file inside the experiment output folder
        p = Path(out_dir)
        candidates = [p / "metrics_final.json", p / "metrics.json", p / "results.json"]
        metrics_obj = None
        for c in candidates:
            if c.exists():
                try:
                    metrics_obj = json.loads(c.read_text())
                except Exception:
                    metrics_obj = None
                if metrics_obj is not None:
                    break

        if metrics_obj is not None:
            # try common locations for eval metrics
            eval_metrics = metrics_obj.get("eval_metrics") or metrics_obj.get("eval") or metrics_obj.get("eval_metrics") or metrics_obj.get("metrics")
            # elapsed time: prefer explicit key, then train_time_s, then train_output log_history
            for k in ("elapsed_s", "elapsed", "duration_s", "duration", "train_time_s", "train_time", "total_time_s"):
                if k in metrics_obj:
                    try:
                        elapsed = float(metrics_obj[k])
                        break
                    except Exception:
                        pass

            if elapsed is None:
                # try train_output.log_history entries
                to = metrics_obj.get("train_output") or metrics_obj.get("train_stats")
                if isinstance(to, dict):
                    lh = to.get("log_history") or []
                    # look for an entry with 'train_runtime' or sum runtimes
                    for entry in reversed(lh):
                        if isinstance(entry, dict) and "train_runtime" in entry:
                            try:
                                elapsed = float(entry.get("train_runtime"))
                                break
                            except Exception:
                                pass

    # fallback: also check top-level keys that might be present in the summary entry itself
    if elapsed is None:
        for k in ("elapsed_s", "elapsed", "duration_s", "duration"):
            if k in e:
                try:
                    elapsed = float(e[k])
                except Exception:
                    pass

    acc = None
    if isinstance(eval_metrics, dict):
        acc = eval_metrics.get("eval_accuracy") or eval_metrics.get("accuracy") or eval_metrics.get("eval_acc") or eval_metrics.get("acc")

    rows.append({
        "optimizer": e.get("optimizer"),
        "seed": e.get("seed"),
        "elapsed_s": elapsed,
        "eval_accuracy": acc,
        "out_dir": out_dir
    })

df = pd.DataFrame(rows)
df.to_csv("experiments_summary_table.csv", index=False)
print(df.groupby("optimizer").agg({"eval_accuracy":["mean","std"], "elapsed_s":"mean"}))

# plot
fig, ax = plt.subplots()
df.boxplot(column="eval_accuracy", by="optimizer", ax=ax)
ax.set_title("Final eval accuracy by optimizer")
ax.set_ylabel("Accuracy")
plt.suptitle("")
plt.savefig("accuracy_by_optimizer.png", bbox_inches="tight")
print("Saved plots: accuracy_by_optimizer.png, CSV: experiments_summary_table.csv")
