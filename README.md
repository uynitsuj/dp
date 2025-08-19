# DP
Diffusion Policy Re-implimentation


## Installation
Clone the repository and initialize submodules:
```bash
git clone --recurse-submodules https://github.com/uynitsuj/dp.git
# Or if already cloned without --recurse-submodules, run:
git submodule update --init --recursive
```
Install the main package and I2RT repo for CAN driver interface using uv:
```bash
cd dp
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
uv sync
source .venv/bin/activate
uv pip install -e .
uv pip install transformers[torch]

```
## Defining training configs and running training
To fine-tune a model on your own data, you need to define configs for data processing and training. We provide example [configs](dp/util/config.py) starting from standard format lerobot, which you can modify for your own dataset.

Dataset normalization statistics are computed and applied to data automatically prior to training.

Now we can kick off training with the following command:
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
- [x] Make lerobot data an acceptable input format (don't need to write a new dataloader, just convert data prior to train, possibly rm data after train)
- [ ] Currently dataloader is hardcoded for single timestep obs, fix this lol