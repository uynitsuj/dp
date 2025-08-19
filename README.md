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
## Launch Train (Lerobot Dataset)
Then run the launch entrypoint script with an appropriate robot config file:
```bash
uv run script/train.py --dataset-cfg.dataset-root /home/justinyu/dp/data/lerobot2dp/uynitsuj/overfit_soup_can_data_20250818 --logging-cfg.log-name 250819_1113 --logging-cfg.output-dir /nfs_us/justinyu/dp --shared-cfg.num-pred-steps 30 --trainer-cfg.epochs 500
```

## Linting
If contributing, please use ruff (automatically installed) for linting (https://docs.astral.sh/ruff/tutorial/#getting-started)
```bash
ruff check # lint
ruff check --fix # lint and fix anything fixable
ruff format # code format
```

## Roadmap/Todos

- [ ] Make DataConfig a ConfigFactory so that it's more extensible to different data formats
- [ ] Make lerobot data an acceptable input format (don't need to write a new dataloader, just convert data prior to train, possibly rm data after train)
- [ ] Currently dataloader is hardcoded for single timestep obs, fix this lol