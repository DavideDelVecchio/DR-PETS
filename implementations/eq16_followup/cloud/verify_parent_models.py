"""Read-only provenance and model-architecture preflight; no training or MPC."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from followup import BRIDGE_DEFAULT, SHARD_SEEDS, check_state_dict, load_parent


def verify_architecture(parent):
    from torch import nn
    from eq16_core import GaussianDynamics

    config = parent["config"]
    if (config.get("ensemble"), config.get("width"), config.get("depth")) != (5, 128, 3):
        raise ValueError("Expected the frozen five-member, width-128, three-hidden-layer study model")
    states, hashes = check_state_dict(parent)
    architectures = []
    for state in states:
        model = GaussianDynamics(18, 6, width=config["width"], depth=config["depth"])
        model.load_state_dict(state)
        layers = [module for module in model.net if isinstance(module, nn.Linear)]
        shape = [(module.in_features, module.out_features) for module in layers]
        if shape != [(24, 128), (128, 128), (128, 128), (128, 36)]:
            raise ValueError("Saved weights do not match the expected three-hidden-layer architecture")
        architectures.append(shape)
    return {
        "training_seed": parent["seed"], "ensemble_members": len(states),
        "hidden_layers": 3, "hidden_width": 128, "linear_shapes": architectures[0],
        "checkpoint_sha256": parent["checkpoint_sha256"], "member_sha256": hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--shard", choices=list(SHARD_SEEDS), required=True)
    parser.add_argument("--stage", choices=["audit", "bridge", "curves"], default="audit")
    parser.add_argument("--all-replicas", action="store_true")
    args = parser.parse_args()
    seeds = BRIDGE_DEFAULT[args.shard] if args.stage == "bridge" and not args.all_replicas else SHARD_SEEDS[args.shard]
    for seed in seeds:
        parent = load_parent(Path(args.results_root).resolve(), seed)
        print("PARENT MODEL VERIFIED:", json.dumps(verify_architecture(parent)), flush=True)


if __name__ == "__main__":
    main()
