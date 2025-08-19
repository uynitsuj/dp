# DP
Personal Diffusion Policy Repo

## Installation
Clone the repository and initialize submodules:
```bash
git clone --recurse-submodules https://github.com/uynitsuj/dp.git
# Or if already cloned without --recurse-submodules, run:
git submodule update --init --recursive
```
Install dependencies and main package using uv:
```bash
cd dp
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv pip install -e .
uv pip install submodules/transformers[torch]

```
## Defining training configs and running training
To fine-tune a model on your own data, you need to define configs for data processing and training. Example [configs](dp/util/config.py) are provided, starting from standard format lerobot dataset, which you can modify and add interfaces for your own dataset format.

Dataset normalization statistics are computed and applied to data automatically prior to training.

We run training with the command:
```bash
uv run scripts/train.py dp_xmi_rby --exp-name=my_experiment
```

## Linting
If contributing, please use ruff (automatically installed) for linting (https://docs.astral.sh/ruff/tutorial/#getting-started)
```bash
ruff check # lint
ruff check --fix # lint and fix anything fixable
ruff format # code format
```

## Roadmap/Todos

- [x] Make DataConfig a ConfigFactory so that it's more extensible to different data formats
- [x] Make lerobot datasets an acceptable input dataset format (no need to write a new dataloader, just convert and cache data mp4->jpg and paraquet->hdf5 prior to train)
- [ ] Add DINOv3 (and/or SigLip) ViT option
- [ ] Develop remote inference wrapper
- [ ] Currently dataloader is hardcoded for single timestep obs, fix this lol