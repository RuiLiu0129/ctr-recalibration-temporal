# Environment — versions behind the paper's software line

The versions below are the ones printed in the paper (§IV-C: "Python 3.12.3,
scikit-learn 1.9.1, PyTorch 2.14.1+cpu, NumPy 2.5.4, SciPy 1.18.1") plus the
remaining pins of the environment that ran every experiment in this repository.

| Component | Version |
|---|---|
| Python | 3.12.3 |
| NumPy | 2.5.4 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| PyTorch | 2.14.1+cpu |
| PyYAML | 6.0.3 |
| matplotlib | 3.11.2 |
| pyarrow | 26.0.0 |

**How they were obtained.** Each version was read directly from the project
virtual environment (`.venv`) in which all runs reported in the paper were
executed — the same interpreter and package set that produced the run logs in
`supplement/logs/` and every result file in this supplement. Nothing was
reconstructed from memory or from a public release catalog; where a build
string is nonstandard (e.g. `torch 2.14.1+cpu`), it is reported exactly as the
environment reports it, because that is the environment that ran the work.

**Pinning.** `requirements.txt` at the repository root pins exactly these
versions, so the environment is reproducible with
`pip install -r requirements.txt`.

**Hardware.** All runs — training, recalibration, timing probes — were
executed on a 2-vCPU virtual machine on an AMD EPYC 9D25 host. All
wall-clock times in the paper (Table III) and in `supplement/logs/` are
from this machine.
