## Publication status and authors’ statement

This repository accompanies the article:

> *Adaptive variable sampling model for performance analysis in high cache-performance computing environments*  
> DOI: https://doi.org/10.1016/j.heliyon.2023.e16777

Heliyon has informed the authors of its decision to retract the article. The authors acknowledge that certain mathematical expressions, derivations, algorithmic descriptions, pseudocode, terminology, figure labels, and explanatory passages in the published manuscript contain errors or ambiguities and require correction or clarification.

However, the authors dispute that these manuscript-level issues, by themselves, invalidate the experimental results or the core scientific conclusions of the study. Accordingly, the authors disagree with the retraction and dispute its grounds.

### Scope and limitations of this repository

This repository is provided to support transparent and independent technical examination of the work. It includes:

- the implementation and analysis code.
- the software dependency information.
- descriptions of the original hardware and operating-system environments.
- the performance-measurement and analysis workflow.
- instructions for collecting new measurements and rerunning the analysis.

The original raw measurement dataset is no longer retained because of the time elapsed since the experiments and the applicable data-retention limitations. Therefore, this repository alone cannot reproduce the exact numerical results reported in the article from the original measurements.

Nevertheless, the repository enables researchers to inspect the implemented computational workflow, evaluate the methodology, and perform an independent replication using newly collected data from the described or compatible computing environments. The authors welcome independent technical review of the code and methodology.

This statement represents the authors’ position.

# Adaptive variable sampling for HPC performance analysis

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead of the `source`
command.

## Collecting the dataset

### NPB workloads

Use NPB Class D and collect the following eight applications:

```text
BT, CG, EP, FT, IS, LU, MG, SP
```

### Experimental systems reported in the paper

Measurements were collected every second with Linux Perf in single and multi
configurations. KNL was evaluated in both cache and flat memory modes.

| System | Operating system | Processor | Cores | Memory | Perf version |
| --- | --- | --- | ---: | ---: | --- |
| KNL | CentOS 7.6.1810 | Intel Xeon Phi 7290 @ 1.50 GHz | 288 | 188 GiB | `3.10.0-1062.18.1.el7.x86_64.debug` |
| SLX | CentOS 7.3.1611 | Intel Xeon Gold 6152 @ 2.10 GHz | 88 | 188 GiB | `3.10.0-1062.12.1.el7.x86_64.debug` |
| EPYC | CentOS 7.8.2003 | AMD EPYC 7451 24-Core Processor | 96 | 251 GiB | `3.10.0-1127.8.2.el7.x86_64.debug` |

The paper also reports a master node with CentOS 7.6.1810, four Intel Xeon
E5-2620 processors (24 total cores), and 31 GiB memory.

### Perf collection

Check which events are supported on the target processor:

```bash
perf list
```

Set the events, output file, and NPB executable. The following example measures
a single executable. For MPI or another execution environment, replace the
command after `perf stat` with the appropriate launcher command.

```bash
EVENTS="event1,event2,event3"
OUTPUT="bt_perf.csv"
NPB_BINARY="./bt.D.x"

perf stat \
  -I 1000 \
  -x, \
  -e "$EVENTS" \
  -o "$OUTPUT" \
  -- "$NPB_BINARY"
```

Collect all eight NPB Class D applications at one-second intervals. Reshape the
output so that each row represents one timestamp and each Perf event is a
numeric column. Add the following metadata and label columns:

| Column | Description |
| --- | --- |
| `Application` | NPB application: BT, CG, EP, FT, IS, LU, MG, or SP |
| `mode` | Platform and execution mode, such as `epyc_perf_single` |
| `cpu` | CPU usage label |
| `memory` | Memory usage label |

Generate the low, middle, and high CPU/memory labels using the workload
calculation and frequency-distribution procedure described in Algorithm 2 and
Figures 3–4 of the paper. The analysis code treats these labels as categorical
values.

By default, the merge script expects the following eight CSV filenames:

```text
cpu_cache_full_multi.csv
cpu_cache_full_single.csv
cpu_epyc_full_multi.csv
cpu_epyc_full_single.csv
cpu_flat_full_multi.csv
cpu_flat_full_single.csv
cpu_slx_full_multi.csv
cpu_slx_full_single.csv
```

If your CSV files use different names, update the `SOURCE_FILES` list near the
top of `1. merge_cpu_full_data.py` before running the merge step. Alternatively,
rename your files to match the default names above.

All eight files must use the same column names and order. Every performance
feature other than `Application`, `mode`, `cpu`, and `memory` must be numeric.
The optional metadata columns `cpu_next`, `memory_next`, and `cpu_score` are
recognized and automatically excluded from model features.

## Running the analysis

Replace `/path/to/npb-data` with the directory containing the eight prepared CSV
files.

### 1. Merge measurements

```bash
python "1. merge_cpu_full_data.py" --data_dir /path/to/npb-data
```

To match the original preprocessing procedure, the script removes the final row
of each `Application` group from every source CSV before merging. The merged
dataset is saved as `/path/to/npb-data/cpu_full_merged.csv`.

### 2. Select variables and split datasets

```bash
python "2. pca_to_csv.py" \
  --csv /path/to/npb-data/cpu_full_merged.csv \
  --out_dir pca_file
```

Add `--verbose` to print details of the PCA and correlation-based variable
selection process.

### 3. Evaluate RBF-SVM models

```bash
python "3. svm_fig.py" --c 0.1  --gamma 2   --arch epyc slx   --target cpu
python "3. svm_fig.py" --c 0.1  --gamma 4   --arch cache flat --target cpu
python "3. svm_fig.py" --c 0.01 --gamma 32  --arch epyc slx   --target memory
python "3. svm_fig.py" --c 1    --gamma 0.5 --arch cache flat --target memory
```

Figures and metrics are written to `fig/`. Evaluation uses
Leave-One-Group-Out cross-validation with `Application` as the group, matching
the paper's leave-one-application-out evaluation.

## Repository files

| File | Purpose |
| --- | --- |
| `1. merge_cpu_full_data.py` | Merge the eight platform/mode CSV files |
| `2. pca_to_csv.py` | PCA-loading variable selection and data splitting |
| `anti_column.py` | PCA-loading and Pearson-correlation selection |
| `3. svm_fig.py` | LOGO RBF-SVM evaluation and result figures |
| `requirements.txt` | Python dependencies |
