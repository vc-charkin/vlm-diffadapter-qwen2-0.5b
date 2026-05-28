from pathlib import Path

from vlm_diffadapter.experiments import load_experiment_recipe


def test_coco_h100_16batch_recipe_is_versioned() -> None:
    recipe = load_experiment_recipe("configs/experiments/coco_h100_datamodule_16batch.yaml")

    assert recipe.run_name == "h100_real_coco_datamodule_16batch"
    assert recipe.output_root == Path("runs")
    assert recipe.model_config == Path("configs/model_h100_real.yaml")
    assert recipe.manifest == Path("data/coco2017_smoke_128/manifest.jsonl")
    assert recipe.adapter_only_checkpoint is True
    assert recipe.device == "cuda"
    assert recipe.use_data_module is True
    assert recipe.val_fraction == 0.1
    assert recipe.max_train_batches == 16
