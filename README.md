# HADA

HADA is an automated algorithm design framework based on MetaBBO, combining MetaBBO for continuous optimization problems.

## Quick Start

### 1. Build Docker Image

```bash
docker build -t hada -f Dockerfile .
```

### 2. Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial commit"
```

### 3. Run Initial Evaluation

Run initial evaluation to establish baseline performance:

```bash
# Run all domains
bash setup_initial.sh

# Or run specific domain only
bash setup_initial.sh bbob_unconstrained
bash setup_initial.sh bbob_constrained
bash setup_initial.sh metabox_mo
```

### 4. Start HADA Generation Loop

```bash
python generate_loop.py --domains bbob_unconstrained --max_generation 100
```

## Supported Domains

- `bbob_unconstrained`: BBOB unconstrained optimization problems
- `bbob_constrained`: BBOB constrained optimization problems
- `metabox_mo`: WFG multi-objective optimization problems

## Project Structure

```
HADA/
├── Dockerfile              # Docker image build file
├── setup_initial.sh        # Initial evaluation script
├── generate_loop.py        # Main generation loop entry
├── metabbo/                # DQN-DE core implementation
│   ├── ec_algorithm.py     # DE optimizer
│   ├── meta_learning.py    # DQN controller
│   └── meta_learning_env.py # RL environment wrapper
├── domains/                # Evaluation logic for each domain
│   ├── bbob_unconstrained/
│   ├── bbob_constrained/
│   └── metabox_mo/
└── outputs/                # Evaluation output directory
```

## Main Parameters

`generate_loop.py` supports the following parameters:

- `--domains`: Domains to optimize (comma-separated)
- `--max_generation`: Maximum number of generations
- `--parent_selection`: Parent selection strategy (score_prop/score_child_prop/random/latest/best)


## Output

Each run generates the following files in the `outputs/` directory:

- `report.json`: Evaluation report
- `predictions.csv`: Prediction results
- `{domain}_return_curves.png`: Training return curves
- `{domain}_loss_curves.png`: Training loss curves