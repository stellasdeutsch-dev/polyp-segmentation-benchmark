# One entry point for every routine task. Run `make help` for the list.
PY      := python3.11
VENV    := .venv
BIN     := $(VENV)/bin
export PYTORCH_ENABLE_MPS_FALLBACK := 1

.PHONY: help setup lock check-env data splits pranet milesial published-maps pranet-run stage3-figures unet-timing overfit train queue eval site-data deploy preview test lint format clean-env

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## create .venv and install the exact pinned versions
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-lock.txt
	$(BIN)/pip install -e . --no-deps

lock:  ## re-resolve from pyproject.toml and rewrite requirements-lock.txt
	$(BIN)/pip install -e ".[dev]"
	@echo "# Exact versions for 'make setup'. Regenerate with 'make lock' after changing pyproject.toml." > requirements-lock.txt
	@echo "# Resolved on macOS $$(sw_vers -productVersion) $$(uname -m), Python $$($(BIN)/python -c 'import platform; print(platform.python_version())')." >> requirements-lock.txt
	$(BIN)/pip freeze --exclude-editable | grep -v '^-e' >> requirements-lock.txt

check-env:  ## print versions, device and memory probe
	$(BIN)/python scripts/check_env.py

data:  ## download and unpack the PraNet benchmark into data/ (~730 MB)
	$(BIN)/python scripts/download_data.py

splits:  ## inventory the data and write data/splits/ and reports/data_inventory.md
	$(BIN)/python scripts/make_splits.py

pranet:  ## fetch PraNet's code (pinned), released weights and the authors' prediction maps
	$(BIN)/python scripts/fetch_pranet.py

milesial:  ## fetch milesial/Pytorch-UNet's model code (pinned), the reference for our U-Net (stage 4)
	$(BIN)/python scripts/fetch_milesial.py

published-maps:  ## score the authors' published maps with our evaluation (stage 3.2)
	$(BIN)/python scripts/eval_published_maps.py

pranet-run:  ## run the released PraNet weights on the Apple GPU and the CPU, and score them (stage 3.3)
	$(BIN)/python scripts/run_pranet_weights.py

stage3-figures:  ## draw our PraNet maps next to the authors' (reports/figures/stage3_maps.png)
	$(BIN)/python scripts/make_stage3_figures.py

unet-timing:  ## time training steps of the from-scratch U-Net and its cheaper variants (stage 4.2)
	$(BIN)/python scripts/time_unet.py

overfit:  ## plumbing check: overfit 32 images, validation Dice should pass 0.9
	$(BIN)/python -m seg.train --config configs/overfit.yaml

train:  ## train one model: make train CONFIG=configs/smp_unet_r34.yaml SET="lr=1e-3 seed=1"
	$(BIN)/python -m seg.train --config $(CONFIG) $(if $(SET),--set $(SET),)

queue:  ## one model through D7: LR search on seed 0, then seeds 1-2; resumable: make queue CONFIG=configs/unet.yaml
	$(BIN)/python scripts/run_queue.py --config $(CONFIG) --rest-minutes 10

eval:  ## score one trained run on the five test sets: make eval RUN=unet_lr3e-4_s0
	$(BIN)/python scripts/eval_run.py --run $(RUN)

site-data:  ## rebuild site/data/journal.json from the journal, the reports and the models
	$(BIN)/python scripts/build_site_data.py

deploy:  ## publish the site to GitHub Pages (snapshot on the gh-pages branch; main is untouched)
	$(BIN)/python scripts/deploy_pages.py

preview:  ## draw training images as the models see them (reports/figures/augmentations.png)
	$(BIN)/python scripts/preview_augmentations.py

test:  ## run the test suite
	$(BIN)/pytest

lint:  ## static checks
	$(BIN)/ruff check src scripts tests research/plot_published.py
	$(BIN)/ruff format --check src scripts tests research/plot_published.py

format:  ## auto-format
	$(BIN)/ruff format src scripts tests research/plot_published.py
	$(BIN)/ruff check --fix src scripts tests research/plot_published.py

clean-env:  ## delete .venv
	rm -rf $(VENV)
