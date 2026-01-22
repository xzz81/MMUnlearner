import os
import pytorch_lightning as pl

from ..common.model_loader import load_model_and_processor
from .eval_module import EvalLightningModule
from .eval_datamodule import EvalDataModule


def run_eval(config):
    model, processor = load_model_and_processor(config)

    lightning_module = EvalLightningModule(
        model=model,
        processor=processor,
        config=config
    )

    data_module = EvalDataModule(config)

    runtime_config = config.get("config", {}).get("runtime", {})
    trainer = pl.Trainer(
        accelerator=runtime_config.get("accelerator", "auto"),
        devices=runtime_config.get("devices", "auto"),
        strategy=runtime_config.get("strategy", "auto"),
        precision=runtime_config.get("precision", "bf16-mixed")
    )

    predictions = trainer.predict(lightning_module, data_module)

    output_file = config.get("eval", {}).get("output_file", "./eval_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    import json
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Evaluation results saved to: {output_file}")
