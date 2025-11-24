import os
import time
import json
import argparse
import random
import inspect
import traceback
import torch
import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# for optimizers - ranger, lamb
try:
    import torch_optimizer as torch_optim
except Exception:
    torch_optim = None

# Default config
DEFAULT_CONFIG = {
    "model": "distilbert-base-uncased",
    "dataset": "imdb",
    "text_col": "text",
    "label_col": "label",
    "optimizer": "ranger",
    "lr": 2e-5,
    "max_train_samples": 200,
    "base_output_dir": "experiments",
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 64,
    "num_train_epochs": 2,
    "weight_decay": 0.0,
    "load_best_model_at_end": False,
    "metric_for_best_model": "accuracy",
    "logging_steps": 50,
    "save_total_limit": 2,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--optimizers", nargs="+", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--base_output_dir", type=str, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=None)
    p.add_argument("--per_device_eval_batch_size", type=int, default=None)
    return p.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_optimizer_fn(optimizer_name, model, lr):
    if optimizer_name is None:
        optimizer_name = "adam"
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == "ranger":
        if torch_optim is None:
            raise RuntimeError("ranger requested but torch-optimizer not installed")
        return torch_optim.Ranger(model.parameters(), lr=lr)
    if optimizer_name == "lamb":
        if torch_optim is None:
            raise RuntimeError("lamb requested but torch-optimizer not installed")
        return torch_optim.Lamb(model.parameters(), lr=lr)
    # fallback
    return torch.optim.Adam(model.parameters(), lr=lr)

def compute_model_stats(model, model_dir):
    num_params = sum(p.numel() for p in model.parameters())
    total_bytes = 0
    if os.path.isdir(model_dir):
        for root, _, files in os.walk(model_dir):
            for f in files:
                try:
                    total_bytes += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return {"num_params": num_params, "model_size_bytes": total_bytes}

def prepare_tokenized_dataset(dataset_name, model_name, text_col="text", label_col="label", max_train_samples=0):
    ds = load_dataset(dataset_name)
    # sub-sample for quick run
    if max_train_samples and max_train_samples > 0:
        n = min(max_train_samples, len(ds["train"]))
        ds["train"] = ds["train"].select(range(n))
        if "test" in ds:
            ds["test"] = ds["test"].select(range(min(1000, len(ds["test"]))))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_tokenize_and_label(examples):
        enc = tokenizer(
            examples[text_col],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        if label_col in examples:
            enc["labels"] = examples[label_col]
        return enc

    tokenized = ds.map(preprocess_tokenize_and_label, batched=True)

    if "labels" not in tokenized["train"].column_names:
        raise ValueError(f"'labels' column not found. Expected dataset column '{label_col}'.")

    tokenized.set_format(
        type="torch",
        columns=[c for c in ["input_ids", "attention_mask", "labels"] if c in tokenized["train"].column_names],
    )

    # determine num_labels
    num_labels = 2
    try:
        feat = ds["train"].features.get(label_col, None)
        if isinstance(feat, ClassLabel):
            num_labels = feat.num_classes
        else:
            unique_labels = set(ds["train"][label_col])
            num_labels = len(unique_labels)
    except Exception:
        pass

    return tokenized, tokenizer, num_labels

def run_single_run(config, optimizer_name, seed):
    out_root = config["base_output_dir"]
    run_name = f"{optimizer_name}_seed{seed}"
    output_dir = os.path.join(out_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare dataset & tokenizer
    try:
        tokenized, tokenizer, num_labels = prepare_tokenized_dataset(
            dataset_name=config["dataset"],
            model_name=config["model"],
            text_col=config["text_col"],
            label_col=config["label_col"],
            max_train_samples=config.get("max_train_samples", 0),
        )
    except Exception as e:
        err = {"error": "dataset_preparation_failed", "exc": repr(e), "trace": traceback.format_exc()}
        with open(os.path.join(output_dir, "metrics_final.json"), "w") as f:
            json.dump({"config": {"optimizer": optimizer_name, "seed": seed}, "error": err}, f, indent=2)
        return {"ok": False, "error": err}

    # create fresh model for each run
    model = AutoModelForSequenceClassification.from_pretrained(config["model"], num_labels=num_labels)

    # build TrainingArguments per run (unique output_dir)
    candidate_args = {
        "output_dir": output_dir,
        "learning_rate": config.get("lr"),
        "per_device_train_batch_size": config.get("per_device_train_batch_size"),
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size"),
        "num_train_epochs": config.get("num_train_epochs"),
        "weight_decay": config.get("weight_decay"),
        "load_best_model_at_end": config.get("load_best_model_at_end"),
        "metric_for_best_model": config.get("metric_for_best_model"),
        "logging_steps": config.get("logging_steps"),
        "save_total_limit": config.get("save_total_limit"),
        "seed": seed,
        "fp16": False,
    }

    sig = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in candidate_args.items() if k in supported and v is not None}
    training_args = TrainingArguments(**filtered_kwargs)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("test"),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # patch optimizer if available
    try:
        opt_obj = get_optimizer_fn(optimizer_name, model, config.get("lr") or 2e-5)
        trainer.optimizer = opt_obj
        def _create_optimizer():
            trainer.optimizer = opt_obj
            return trainer.optimizer
        trainer.create_optimizer = _create_optimizer
        optimizer_status = {"used": True, "name": optimizer_name}
    except Exception as e:
        optimizer_status = {"used": False, "name": optimizer_name, "reason": repr(e)}
        print(f"[warning] couldn't create optimizer {optimizer_name}: {e}")

    # training + evaluation
    start = time.time()
    try:
        train_output = trainer.train()
        train_time_s = time.time() - start
        log_history = trainer.state.log_history
        try:
            metrics = trainer.evaluate()
        except Exception as e_eval:
            metrics = {"error": "evaluation_failed", "exc": repr(e_eval), "trace": traceback.format_exc()}
    except Exception as e_train:
        train_time_s = time.time() - start
        err = {"error": "training_failed", "exc": repr(e_train), "trace": traceback.format_exc()}
        # save error and return
        with open(os.path.join(output_dir, "metrics_final.json"), "w") as f:
            json.dump({"config": {"optimizer": optimizer_name, "seed": seed}, "train_error": err}, f, indent=2)
        # also save stderr trace
        with open(os.path.join(output_dir, "stderr.txt"), "w") as f:
            f.write(traceback.format_exc())
        return {"ok": False, "error": err}

    # save model
    trainer.save_model(output_dir)

    # collect model stats
    model_stats = compute_model_stats(model, output_dir)

    # finalize
    final_obj = {
        "config": {
            "model": config["model"],
            "dataset": config["dataset"],
            "optimizer_requested": optimizer_name,
            "optimizer_used": optimizer_status,
            "lr": config.get("lr"),
            "seed": seed,
            "num_train_samples": len(tokenized["train"]),
            "num_eval_samples": len(tokenized.get("test", [])) if tokenized.get("test") else None,
            "num_train_epochs": config.get("num_train_epochs"),
            "per_device_train_batch_size": config.get("per_device_train_batch_size"),
        },
        "train_time_s": train_time_s,
        "train_output": {
            "train_loss": float(train_output.training_loss) if hasattr(train_output, "training_loss") else None,
            "log_history": log_history,
        },
        "eval_metrics": metrics,
        "model_stats": model_stats,
    }

    with open(os.path.join(output_dir, "metrics_final.json"), "w") as f:
        json.dump(final_obj, f, indent=2)

    print(f"[done] {optimizer_name} seed={seed} -> eval: {metrics}")
    return {"ok": True, "metrics": final_obj, "output_dir": output_dir}

def main():
    args = parse_args()
    # merging defaults and args
    config = DEFAULT_CONFIG.copy()
    if args.model: config["model"] = args.model
    if args.dataset: config["dataset"] = args.dataset
    if args.lr is not None: config["lr"] = args.lr
    if args.max_train_samples is not None: config["max_train_samples"] = args.max_train_samples
    if args.base_output_dir: config["base_output_dir"] = args.base_output_dir
    if args.num_train_epochs is not None: config["num_train_epochs"] = args.num_train_epochs
    if args.per_device_train_batch_size is not None: config["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.per_device_eval_batch_size is not None: config["per_device_eval_batch_size"] = args.per_device_eval_batch_size

    # choose optimizers and seeds
    if args.optimizers:
        optimizers = args.optimizers
    else:
        optimizers = ["adam", "ranger", "lamb"]

    if args.seeds:
        seeds = args.seeds
    else:
        seeds = [42, 2025, 11]

    base_out = config["base_output_dir"]
    os.makedirs(base_out, exist_ok=True)

    summary = []
    
    for opt in optimizers:
        for seed in seeds:
            set_seed(seed)
            if opt.lower() in ("ranger", "lamb") and torch_optim is None:
                msg = f"Skipping {opt} because torch-optimizer is not installed. Install with: pip install torch-optimizer"
                print(msg)
                entry = {"optimizer": opt, "seed": seed, "skipped": True, "reason": "torch-optimizer not installed"}
                # write a small json so analyzer knows it was skipped
                out_dir = os.path.join(base_out, f"{opt}_seed{seed}")
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "metrics_final.json"), "w") as f:
                    json.dump({"skipped": True, "reason": "torch-optimizer not installed", "optimizer": opt, "seed": seed}, f, indent=2)
                summary.append(entry)
                continue

            try:
                res = run_single_run(config, opt.lower(), seed)
                entry = {"optimizer": opt, "seed": seed, "ok": res.get("ok", False), "out_dir": res.get("output_dir", None)}
                if not res.get("ok", False):
                    entry["error"] = res.get("error")
                summary.append(entry)
            except Exception as e:
                err = {"exc": repr(e), "trace": traceback.format_exc()}
                print(f"[fatal] run failed for {opt} seed {seed}: {e}")
                out_dir = os.path.join(base_out, f"{opt}_seed{seed}")
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "metrics_final.json"), "w") as f:
                    json.dump({"error": "unexpected_exception", "exc": repr(e)}, f, indent=2)
                summary.append({"optimizer": opt, "seed": seed, "ok": False, "error": err, "out_dir": out_dir})

            # small delay to release some memory
            time.sleep(1)

    # save summary
    with open(os.path.join(base_out, "experiments_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll runs completed.")

if __name__ == "__main__":
    main()
