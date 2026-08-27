"""Train a DeepTDA collective variable on the frame-level dataset produced by
``prepare_deeptda_data.py`` (reactive/non-reactive) or
``prepare_deeptda_data_ld.py`` (two simulations, e.g. L/D enantiomers).

mlcolvar's ``DeepTDA``/``TDALoss`` (checked against the installed source,
mlcolvar 1.3.1) only ever reads the ``"data"`` and ``"labels"`` keys of a
batch - any ``"weights"`` key is silently ignored. So the per-frame
frame weight computed during data prep can't be handed to the loss directly;
instead each class is importance-resampled (with replacement, probability
proportional to its own weights) into a balanced, effectively-unweighted set
of ``n_per_class`` frames per class before building the DictDataset. This
also fixes the class imbalance (e.g. reactive/non-reactive) that's typical of
TIS path ensembles.
"""

import shutil
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
import lightning
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from mlcolvar.cvs import DeepTDA
from mlcolvar.data import DictDataset, DictModule
from mlcolvar.utils.trainer import MetricsCallback

from inftools.tistools.shap_analysis import _check_overwrite


def _weighted_pearson(x, y, w):
    """Weighted Pearson correlation coefficient between x and y."""
    wx = np.average(x, weights=w)
    wy = np.average(y, weights=w)
    cov = np.average((x - wx) * (y - wy), weights=w)
    varx = np.average((x - wx) ** 2, weights=w)
    vary = np.average((y - wy) ** 2, weights=w)
    return cov / np.sqrt(varx * vary)


def _weighted_resample(X, y, w, n_per_class, rng):
    """Importance-resample each class separately to size n_per_class,
    drawing frame i of class c with probability w[i] / sum(w[class c])."""
    X_out, y_out = [], []
    for cls in (0.0, 1.0):
        idx = np.nonzero(y == cls)[0]
        if len(idx) == 0:
            raise ValueError(f"No frames found for class {cls}.")
        p = w[idx]
        if not np.any(p > 0):
            raise ValueError(f"All weights are zero for class {cls}.")
        p = p / p.sum()
        chosen = rng.choice(idx, size=n_per_class, replace=True, p=p)
        X_out.append(X[chosen])
        y_out.append(np.full(n_per_class, cls))
    return np.concatenate(X_out), np.concatenate(y_out)


def train_deeptda(
    npz: Annotated[str, typer.Option("-npz", help="Dataset .npz from prepare-deeptda-data.")] = "deeptda_ld_dataset.npz",
    out_dir: Annotated[str, typer.Option("-out-dir", help="Output folder for the trained model and diagnostic plots.")] = "deeptda_model",
    n_per_class: Annotated[int, typer.Option("-n-per-class", help="Importance-resampled frames per class.")] = 500000,
    n_cvs: Annotated[int, typer.Option("-n-cvs", help="Number of CVs to learn.")] = 1,
    hidden: Annotated[str, typer.Option("-hidden", help="Comma-separated hidden layer sizes.")] = "32,16",
    target_centers: Annotated[str, typer.Option("-target-centers", help="Comma-separated Gaussian target centers, one per state.")] = "-.5,.5",
    target_sigmas: Annotated[str, typer.Option("-target-sigmas", help="Comma-separated Gaussian target sigmas, one per state.")] = "0.2,0.2",
    val_frac: Annotated[float, typer.Option("-val-frac", help="Fraction of the resampled set held out for validation.")] = 0.2,
    batch_size: Annotated[int, typer.Option("-batch-size", help="Training batch size.")] = 4096,
    max_epochs: Annotated[int, typer.Option("-max-epochs", help="Maximum number of training epochs.")] = 10000,
    patience: Annotated[int, typer.Option("-patience", help="Stop after this many epochs with no validation-loss improvement; the best (not final) epoch's weights are saved.")] = 1000,
    lr: Annotated[float, typer.Option("-lr", help="Adam learning rate.")] = 1e-3,
    seed: Annotated[int, typer.Option("-seed", help="Random seed for resampling and training.")] = 42,
    class_names: Annotated[str, typer.Option("-class-names", help="Comma-separated names for class 0,1 used in prints and plot legends, e.g. 'non-reactive,reactive' (prepare-deeptda-data) or 'L,D' (prepare-deeptda-data-ld).")] = "non-reactive,reactive",
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of files.")] = False,
):
    """Train a 2-state DeepTDA CV on a prepare-deeptda-data(-ld) dataset."""
    name0, name1 = (s.strip() for s in class_names.split(","))
    out_path = Path(out_dir)
    model_path = out_path / "deeptda_model.ptc"
    loss_plot_path = out_path / "loss_curve.png"
    hist_plot_path = out_path / "cv_histogram.png"
    op_corr_plot_path = out_path / "cv_vs_op.png"
    input_corr_plot_path = out_path / "cv_vs_inputs.png"
    for p in [model_path, loss_plot_path, hist_plot_path, op_corr_plot_path, input_corr_plot_path]:
        _check_overwrite(str(p), overw)
    out_path.mkdir(parents=True, exist_ok=True)

    lightning.seed_everything(seed)
    rng = np.random.default_rng(seed)

    raw = np.load(npz, allow_pickle=True)
    X, y, w, op = raw["X"], raw["y"], raw["w"], raw["op"]
    cv_names = list(raw["cv_names"])
    n_1, n_0 = int(np.sum(y == 1.0)), int(np.sum(y == 0.0))
    print(f"Loaded {len(y)} frames ({n_1} {name1} / {n_0} {name0}), {len(cv_names)} CVs.")

    X_res, y_res = _weighted_resample(X, y, w, n_per_class, rng)
    print(f"Resampled to {len(y_res)} frames ({n_per_class} per class).")

    dataset = DictDataset({"data": X_res, "labels": y_res})
    datamodule = DictModule(
        dataset,
        lengths=[1.0 - val_frac, val_frac],
        batch_size=batch_size,
    )

    layers = [X.shape[1], *[int(h) for h in hidden.split(",")], n_cvs]
    centers = [float(c) for c in target_centers.split(",")]
    sigmas = [float(s) for s in target_sigmas.split(",")]

    model = DeepTDA(
        n_states=2,
        n_cvs=n_cvs,
        target_centers=centers,
        target_sigmas=sigmas,
        layers=layers,
        options={"optimizer": {"lr": lr}},
    )

    # Train loss can keep falling on the resampled set while valid loss rises
    # (overfitting to that resampling) - track the best valid-loss epoch and
    # export those weights rather than whatever the final epoch produced.
    ckpt_dir = out_path / "checkpoints"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), filename="best", monitor="valid_loss", mode="min", save_top_k=1
    )
    early_stop_cb = EarlyStopping(monitor="valid_loss", mode="min", patience=patience)

    metrics = MetricsCallback()
    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=10,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        callbacks=[metrics, checkpoint_cb, early_stop_cb],
    )
    trainer.fit(model, datamodule)

    if checkpoint_cb.best_model_path and checkpoint_cb.best_model_score is not None:
        best_epoch_loss = checkpoint_cb.best_model_score.item()
        print(
            f"Restoring best-valid-loss checkpoint "
            f"(valid_loss={best_epoch_loss:.4f}) from {checkpoint_cb.best_model_path}"
        )
        state = torch.load(checkpoint_cb.best_model_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
    else:
        print("WARNING: no checkpoint was saved; exporting the final epoch's weights.")
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    model.eval()
    traced = model.to_torchscript(
        file_path=str(model_path), method="trace", example_inputs=model.example_input_array
    )
    print(f"Saved trained model to {model_path}")

    epochs = metrics.metrics["epoch"]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, metrics.metrics["train_loss_epoch"], label="train")
    if "valid_loss" in metrics.metrics:
        plt.plot(epochs, metrics.metrics["valid_loss"], label="valid")
    plt.xlabel("epoch")
    plt.ylabel("TDA loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close("all")
    print(f"Saved loss curve to {loss_plot_path}")

    with torch.no_grad():
        s = model(torch.Tensor(X)).numpy().reshape(-1)
    plt.figure(figsize=(6, 4))
    plt.hist(s[y == 0.0], bins=80, weights=w[y == 0.0], density=True, alpha=0.6, label=name0)
    plt.hist(s[y == 1.0], bins=80, weights=w[y == 1.0], density=True, alpha=0.6, label=name1)
    plt.xlabel("DeepTDA CV")
    plt.ylabel("weighted density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_plot_path, dpi=150)
    plt.close("all")
    print(f"Saved CV histogram (full, path-weighted dataset) to {hist_plot_path}")

    # Quick sanity check: does the learned CV track the original order
    # parameter, on the true (path-weighted) ensemble rather than the
    # class-balanced resampled training set?
    r = _weighted_pearson(s, op, w)
    plt.figure(figsize=(6, 5))
    plt.hist2d(op, s, bins=80, weights=w, cmap="viridis")
    plt.xlabel("order parameter (op)")
    plt.ylabel("DeepTDA CV")
    plt.title(f"weighted Pearson r = {r:.3f}")
    plt.colorbar(label="weighted density")
    plt.tight_layout()
    plt.savefig(op_corr_plot_path, dpi=150)
    plt.close("all")
    print(f"Saved CV-vs-op correlation plot to {op_corr_plot_path} (weighted r = {r:.3f})")

    # Which physical input CVs is the learned coordinate actually built from?
    # (op above is the pre-existing order parameter, not necessarily one of
    # the model's inputs - this checks against every column that went into X.)
    input_corrs = np.array([_weighted_pearson(s, X[:, j], w) for j in range(X.shape[1])])
    order = np.argsort(-np.abs(input_corrs))
    print("Weighted Pearson r of learned CV vs. each input CV:")
    for j in order:
        print(f"  {cv_names[j]:>20s}: r = {input_corrs[j]:+.3f}")

    plt.figure(figsize=(6, 0.4 * len(cv_names) + 1))
    plt.barh([cv_names[j] for j in order][::-1], input_corrs[order][::-1])
    plt.xlabel("weighted Pearson r (learned CV vs. input CV)")
    plt.axvline(0, color="k", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(input_corr_plot_path, dpi=150)
    plt.close("all")
    print(f"Saved CV-vs-input-CV correlation plot to {input_corr_plot_path}")


if __name__ == "__main__":
    typer.run(train_deeptda)
