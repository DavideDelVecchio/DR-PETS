# Makefile for DR-PETS workflow

.PHONY: all collect dynamics density debug compare clean

all: collect dynamics density compare

collect:
	python collect_dataset.py

dynamics:
	python train_dynamics_and_export.py --device cuda

density:
	python train_density_model.py --device cuda

debug:
	python debug_pets_rollout.py

compare:
	python scripts/compare_pets_drpets.py

clean:
	rm -f state_action_dataset.pt
	rm -f checkpoints/*.pt
	rm -rf __pycache__
