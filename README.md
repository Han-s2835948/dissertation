# Hard-constraint conditional diffusion on MNIST

This repository contains the code used for the dissertation **Representing and
Enforcing Hard Constraints in Conditional Diffusion Models: A Study of Targeted
MNIST Generation**.

The project trains a continuous-time unconditional score model on MNIST and
then freezes it.  Two measurable target sets for digit 7 are considered:

- a classifier-defined set, where a trained classifier predicts class 7; and
- a calculable geometric set, based on width, slope, connected components,
  holes, and lower-stroke structure.

For each set, unconditional reverse-diffusion trajectories are labelled by
their terminal set membership.  A time-dependent network `h_phi` is trained
with the martingale-loss (CDG-ML) objective.  Its spatial log-gradient is then
added to the reverse drift during conditional sampling.

## Repository layout

```text
configs/       Training settings and the final geometric-set thresholds
mnist_cdg/     Models, SDE, target sets, training, and sampling code
experiments/   Formal eta study and the analyses reported in the dissertation
tests/         Small CPU checks for tensor shapes and numerical finiteness
```

The `outputs/` and `outputs_formal/` directories are created during execution
and are intentionally not tracked by Git.  MNIST is downloaded by torchvision
on first use.

## Environment

The formal experiments were run with:

- Python 3.10.20
- PyTorch 2.13.0+cu130
- torchvision 0.28.0+cu130
- NumPy 2.2.6
- SciPy 1.15.3
- PyYAML 6.0.3
- Pillow 12.2.0
- tqdm 4.70.0
- TensorBoard 2.21.0

The exact CUDA build of PyTorch depends on the local GPU driver.  Install the
appropriate PyTorch build first, then install the remaining packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

Run commands from the repository root.  A GPU is strongly recommended for
training and formal sampling, although the small tests can run on CPU.

## Quick checks

```powershell
python -m mnist_cdg.env_check
python -m unittest discover -s tests -v
```

A short score-training smoke test can also be used to check the MNIST download
and backward pass:

```powershell
python -m mnist_cdg.train_score --config configs/mnist_vp.yaml --epochs 1 --max-batches 1
```

The resulting checkpoint is only a software check and should not be used for
the dissertation experiments.

## Reproducing the main experiment

The commands below follow the order used in the dissertation.  They use the
formal configuration, which writes results under `outputs_formal/mnist/`.

### 1. Train the unconditional score model

```powershell
python -m mnist_cdg.train_score --config configs/mnist_vp_formal.yaml
python -m mnist_cdg.sample_unconditional `
  --config configs/mnist_vp_formal.yaml `
  --checkpoint outputs_formal/mnist/score/latest.pt `
  --samples 64 --steps 500
```

### 2. Train the classifier used to define the learned set

```powershell
python -m mnist_cdg.train_classifier --config configs/mnist_vp_formal.yaml
```

### 3. Recreate the calculable geometric set

```powershell
python -m mnist_cdg.evaluate_geometric_constraint `
  --data-dir data `
  --output outputs_formal/mnist/geometric_constraint_v2
```

The fixed thresholds reported in the dissertation are also stored in
`configs/geometry_v2_thresholds.json` so that the sampling experiment does not
depend on rerunning the threshold search.

### 4. Generate off-policy trajectories

Classifier-defined terminal event:

```powershell
python -m mnist_cdg.generate_trajectories `
  --config configs/mnist_vp_formal.yaml `
  --score outputs_formal/mnist/score/latest.pt `
  --classifier outputs_formal/mnist/classifier/latest.pt `
  --constraint classifier --target-class 7 --samples 50000 `
  --output outputs_formal/mnist/trajectories/classifier_7
```

Geometric terminal event:

```powershell
python -m mnist_cdg.generate_trajectories `
  --config configs/mnist_vp_formal.yaml `
  --score outputs_formal/mnist/score/latest.pt `
  --constraint geometry `
  --thresholds configs/geometry_v2_thresholds.json `
  --samples 50000 `
  --output outputs_formal/mnist/trajectories/geometry_7
```

### 5. Train the two conditioning networks

```powershell
python -m mnist_cdg.train_h `
  --config configs/mnist_vp_formal.yaml `
  --trajectories outputs_formal/mnist/trajectories/classifier_7 `
  --name classifier_7

python -m mnist_cdg.train_h `
  --config configs/mnist_vp_formal.yaml `
  --trajectories outputs_formal/mnist/trajectories/geometry_7 `
  --name geometry_7
```

### 6. Run the paired multi-seed eta studies

Classifier-defined set:

```powershell
python -m experiments.mnist_formal_eta_study `
  --config configs/mnist_vp_formal.yaml `
  --constraint classifier `
  --h-checkpoint outputs_formal/mnist/h/classifier_7.pt `
  --etas 0 1 4 16 64 128 256 512 `
  --output-dir outputs_formal/mnist/formal_eta_multiseed_v1
```

Calculable geometric set:

```powershell
python -m experiments.mnist_formal_eta_study `
  --config configs/mnist_vp_formal.yaml `
  --constraint geometry `
  --thresholds configs/geometry_v2_thresholds.json `
  --h-checkpoint outputs_formal/mnist/h/geometry_7.pt `
  --etas 0 1 16 64 256 512 `
  --output-dir outputs_formal/mnist/formal_eta_geometry_multiseed_v1
```

Each setting uses seeds 8042, 9042, and 10042, with 1,024 images per seed and
500 Euler--Maruyama steps.  Different eta values within a seed reuse the same
random inputs, which permits pathwise comparison with the unconditional sample.

## Analysis scripts

The following programs produce the additional checks used in the results
chapter:

| Program | Purpose |
|---|---|
| `experiments.mnist_independent_classifier_audit` | Trains a second classifier and evaluates generated images |
| `experiments.mnist_h_independent_audit` | Evaluates `h_phi` and its gradient along new trajectories |
| `experiments.mnist_quality_diversity_audit` | Measures feature diversity and distance to real sevens |
| `experiments.mnist_failure_transition_audit` | Groups guided failures by their unconditional class |
| `experiments.mnist_geometry_component_audit` | Checks which geometric conditions fail |
| `experiments.mnist_make_formal_plots` | Creates the combined diagnostic plots |

Use `python -m <module> --help` to see the expected checkpoints and result paths.
The geometric development programs V1--V7 are retained in `mnist_cdg/` because
their precision--recall results are discussed in the methodology chapter.

## CDG-MCL pilot

`build_mcl_targets.py`, `train_q_targets.py`, and `train_q.py` contain the
martingale-covariation pilot.  It is included for completeness but was not used
in the formal eta study because its finite-step targets did not provide a
useful gradient estimator in this implementation.

## Checkpoints

The trained models used in the dissertation are provided with the
[`v1.0.0` release](https://github.com/Han-s2835948/dissertation/releases/tag/v1.0.0)
as `mnist_pretrained_models_v1.0.0.zip`.  Extract the archive into the repository
root to recreate the expected paths under `outputs_formal/mnist/`.

The same release also provides `mnist_reported_results_v1.0.0.zip`, containing
the numerical summaries and PDF figures reported in the dissertation.  SHA-256
checksums are included in both archives.  MNIST is still downloaded by
torchvision on first use, while intermediate trajectories and complete sample
tensors can be regenerated with the commands above.

## Reference

The conditioning method is based on:

Guo, W., Tang, W. and Xu, R. (2026), *Conditional Diffusion Guidance under Hard
Constraint*, arXiv:2602.05533v2.

The MNIST formulation, the two target-set definitions, and the experiment
orchestration in this repository were developed for the dissertation project.
