import argparse
from utils.config_loader import load_config
from src.finetune_phase.run import run_finetune
from src.unlearn_phase.run import run_unlearn, run_merger_only_unlearn
from src.eval_phase.run import run_eval


def main(config_dir: str):
    config = load_config(config_dir)

    if config["config"]["finetune_phase"]["enable"]:
        run_finetune(config)

    if config["config"]["unlearn_phase"]["enable"]:
        unlearn_method = config["config"]["unlearn_phase"].get("method", "manifold")
        if unlearn_method == "merger_only":
            run_merger_only_unlearn(config)
        else:
            run_unlearn(config)

    if config["config"]["eval_phase"]["enable"]:
        run_eval(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config", help="配置目录路径")
    args = parser.parse_args()
    main(args.config)
