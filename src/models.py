"""Models: Logistic Regression (sklearn), DeepFM and DCN-v2 (PyTorch).

All models share one interface:

    model.fit(X, y)            X: (n, n_fields) integer global vocab indices
                               (ndarray or read-only memmap; batches are
                               streamed, the full matrix is never copied)
    model.predict_proba(X)     -> (n,) float64 click probabilities

The global index space comes from ``data_prep.VocabEncoder``: every field's
local indices are offset by the field's offset, so column j of X only ever
contains indices in [offset_j, offset_j + dim_j).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier


# ---------------------------------------------------------------------------
# Logistic Regression on one-hot encoded categorical fields
# ---------------------------------------------------------------------------

def indices_to_sparse(X: np.ndarray, total_vocab: int) -> sp.csr_matrix:
    """One-hot encode an (n, n_fields) global-index matrix into a CSR matrix."""
    n, n_fields = X.shape
    rows = np.repeat(np.arange(n), n_fields)
    cols = X.ravel()
    data = np.ones(n * n_fields, dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, total_vocab))


class LogisticRegressionModel:
    """Logistic regression over field one-hots, trained with SGD.

    Solver choice (Plan A): sklearn ``SGDClassifier(loss="log_loss")``
    trained with ``partial_fit`` over fixed-size batches streamed from the
    (possibly memmap-backed) encoded train array.

    Why not lbfgs (the original scaffold default): lbfgs needs the full
    design materialised and, in the smoke run, hit ``max_iter=300`` without
    converging even on 200k rows. SGD training demonstrably finishes -- a
    fixed number of epochs over the capped training set -- with O(batch)
    memory. Regularisation mirrors the old ``C`` semantics: sklearn's
    LogisticRegression minimises ``C * sum(loss) + 0.5||w||^2`` while
    SGDClassifier minimises ``mean(loss) + alpha * 0.5||w||^2``, hence
    ``alpha = 1 / (C * n_train)``. ``average=True`` (averaged SGD) gives
    stabler, better-calibrated probabilities -- relevant for a calibration
    study. Prediction is also batched.

    Schedule: the config uses ``learning_rate="adaptive", eta0=0.01``.
    sklearn's ``"optimal"`` heuristic was tried first and mis-scales at
    this alpha: the predicted probabilities collapse toward 0 (ranking
    survives, ECE ends up at the base rate). Measured on the temporal
    gap-0 bucket at 500k train rows: optimal -> AUC 0.724 / logloss 1.96
    / ECE 0.170; adaptive eta0=0.01 -> AUC 0.724 / logloss 0.423 /
    ECE 0.010.
    """

    _PREDICT_BATCH = 65536

    def __init__(self, total_vocab: int, params: dict):
        self.total_vocab = total_vocab
        self.params = params
        self.clf: SGDClassifier | None = None

    def fit(self, X, y) -> "LogisticRegressionModel":
        n = len(y)
        C = float(self.params.get("C", 1.0))
        alpha = 1.0 / (C * n) if C > 0 else 0.0
        seed = int(self.params.get("seed", 42))
        kwargs = dict(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            fit_intercept=True,
            max_iter=1,          # epochs are driven by the partial_fit loop
            tol=None,
            learning_rate=str(self.params.get("learning_rate", "optimal")),
            average=bool(self.params.get("average", True)),
            random_state=seed,
        )
        if "eta0" in self.params and kwargs["learning_rate"] != "optimal":
            kwargs["eta0"] = float(self.params["eta0"])
        clf = SGDClassifier(**kwargs)
        batch = int(self.params.get("batch_size", 16384))
        epochs = int(self.params.get("epochs", 3))
        classes = np.array([0, 1])
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            perm = rng.permutation(n)
            for start in range(0, n, batch):
                idx = perm[start:start + batch]
                Xb = indices_to_sparse(np.asarray(X[idx]), self.total_vocab)
                clf.partial_fit(Xb, np.asarray(y[idx]), classes=classes)
        self.clf = clf
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("LogisticRegressionModel.predict_proba before fit")
        out = np.empty(len(X), dtype=np.float64)
        for start in range(0, len(X), self._PREDICT_BATCH):
            Xb = np.asarray(X[start:start + self._PREDICT_BATCH])
            out[start:start + len(Xb)] = self.clf.predict_proba(
                indices_to_sparse(Xb, self.total_vocab))[:, 1]
        return out


# ---------------------------------------------------------------------------
# Shared PyTorch base for the deep models
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class _TorchCTRBase(nn.Module):
    """Embedding + shared training loop for the deep CTR models."""

    def __init__(self, field_dims: list[int], params: dict, device: str = "auto"):
        super().__init__()
        self.field_dims = field_dims
        self.n_fields = len(field_dims)
        self.emb_dim = int(params.get("emb_dim", 16))
        self.params = params
        self.device = _resolve_device(device)
        offsets = np.cumsum([0] + field_dims[:-1])
        self.register_buffer("field_offsets", torch.tensor(offsets, dtype=torch.long))
        total_vocab = int(sum(field_dims))
        self.embedding = nn.Embedding(total_vocab, self.emb_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        # One-dimensional embedding for the first-order (linear) term.
        self.linear_embedding = nn.Embedding(total_vocab, 1)
        nn.init.normal_(self.linear_embedding.weight, mean=0.0, std=0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    # -- subclasses implement forward(X) -> logits ---------------------------
    def forward(self, X: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _embed(self, X: torch.Tensor) -> torch.Tensor:
        """X: (batch, n_fields) global indices -> (batch, n_fields, emb_dim)."""
        return self.embedding(X)

    def _linear_term(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear_embedding(X).sum(dim=1).squeeze(1) + self.bias

    def fit(self, X, y) -> "_TorchCTRBase":
        """Stream shuffled batches from X (ndarray or read-only memmap).

        Only one batch of rows is materialised at a time, so the encoded
        training set can live on disk. Batch order per epoch is a seeded
        permutation (deterministic given the config seed), replacing the
        legacy in-memory TensorDataset/DataLoader path.
        """
        params = self.params
        self.to(self.device)
        n = len(y)
        y_all = np.asarray(y, dtype=np.float32)
        batch = int(params.get("batch_size", 2048))
        rng = np.random.default_rng(int(params.get("seed", 42)))
        opt = torch.optim.Adam(
            self.parameters(),
            lr=float(params.get("lr", 0.001)),
            weight_decay=float(params.get("weight_decay", 1e-5)),
        )
        loss_fn = nn.BCEWithLogitsLoss()
        self.train()
        for _ in range(int(params.get("epochs", 3))):
            perm = rng.permutation(n)
            for start in range(0, n, batch):
                idx = perm[start:start + batch]
                xb = torch.tensor(np.asarray(X[idx], dtype=np.int64), device=self.device)
                yb = torch.tensor(y_all[idx], device=self.device)
                opt.zero_grad()
                loss = loss_fn(self.forward(xb), yb)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.eval()
        batch = int(self.params.get("batch_size", 2048))
        probs = []
        for start in range(0, len(X), batch):
            xb = torch.tensor(
                np.asarray(X[start:start + batch], dtype=np.int64), device=self.device)
            probs.append(torch.sigmoid(self.forward(xb)).cpu().numpy())
        return np.concatenate(probs).astype(np.float64)


def _mlp_tower(input_dim: int, hidden: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = input_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# DeepFM
# ---------------------------------------------------------------------------

class DeepFMModel(_TorchCTRBase):
    """DeepFM: FM second-order interactions + MLP tower + linear term."""

    def __init__(self, field_dims: list[int], params: dict, device: str = "auto"):
        super().__init__(field_dims, params, device)
        self.dnn = _mlp_tower(
            self.n_fields * self.emb_dim,
            list(params.get("mlp", [128, 64])),
            float(params.get("dropout", 0.2)),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        emb = self._embed(X)                                   # (B, F, D)
        sum_emb = emb.sum(dim=1)                               # (B, D)
        fm = 0.5 * (sum_emb ** 2 - (emb ** 2).sum(dim=1)).sum(dim=1)
        dnn_out = self.dnn(emb.reshape(emb.size(0), -1)).squeeze(1)
        return fm + dnn_out + self._linear_term(X)


# ---------------------------------------------------------------------------
# DCN-v2
# ---------------------------------------------------------------------------

class _CrossNetV2(nn.Module):
    """DCN-v2 cross network with full-rank cross matrices."""

    def __init__(self, input_dim: int, n_layers: int):
        super().__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim, input_dim) * 0.01) for _ in range(n_layers)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(n_layers)]
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for W, b in zip(self.weights, self.biases):
            x = x0 * (x @ W + b) + x
        return x


class DCNv2Model(_TorchCTRBase):
    """DCN-v2 (parallel variant): cross network + MLP tower + linear term."""

    def __init__(self, field_dims: list[int], params: dict, device: str = "auto"):
        super().__init__(field_dims, params, device)
        d = self.n_fields * self.emb_dim
        self.cross = _CrossNetV2(d, int(params.get("cross_layers", 3)))
        self.dnn = _mlp_tower(
            d,
            list(params.get("mlp", [128, 64])),
            float(params.get("dropout", 0.2)),
        )
        self.combine = nn.Linear(2, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        emb = self._embed(X).reshape(X.size(0), -1)            # (B, F*D)
        crossed = self.cross(emb)
        cross_out = crossed.sum(dim=1, keepdim=True)           # (B, 1) pooled summary
        dnn_out = self.dnn(emb)                                # (B, 1)
        combined = self.combine(torch.cat([cross_out, dnn_out], dim=1)).squeeze(1)
        return combined + self._linear_term(X)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_NAMES = ("lr", "deepfm", "dcnv2")


def build_model(name: str, encoder, cfg: dict):
    """Construct an unfitted model from the config's per-model params."""
    params = dict(cfg["models"][name])
    params.setdefault("seed", int(cfg.get("seed", 42)))
    device = cfg.get("train", {}).get("device", "auto")
    if name == "lr":
        return LogisticRegressionModel(encoder.total_vocab, params)
    if name == "deepfm":
        return DeepFMModel(encoder.field_dims, params, device)
    if name == "dcnv2":
        return DCNv2Model(encoder.field_dims, params, device)
    raise ValueError(f"unknown model name: {name!r} (expected one of {MODEL_NAMES})")
