from pathlib import Path

from vlm_diffadapter.experiments import load_experiment_recipe


def test_coco_h100_16batch_seed_sweep_recipes_are_versioned() -> None:
    expected = {
        "configs/experiments/coco_h100_datamodule_16batch_seed45.yaml": 45,
        "configs/experiments/coco_h100_datamodule_16batch_seed46.yaml": 46,
        "configs/experiments/coco_h100_datamodule_16batch_seed47.yaml": 47,
    }

    for recipe_path, seed in expected.items():
        recipe = load_experiment_recipe(recipe_path)
        assert recipe.run_name == f"h100_real_coco_datamodule_16batch_seed{seed}"
        assert recipe.output_root == Path("runs")
        assert recipe.model_config == Path("configs/model_h100_real.yaml")
        assert recipe.manifest == Path("data/coco2017_smoke_128/manifest.jsonl")
        assert recipe.seed == seed
        assert recipe.adapter_only_checkpoint is True
        assert recipe.device == "cuda"
        assert recipe.use_data_module is True
        assert recipe.val_fraction == 0.1
        assert recipe.max_train_batches == 16
        assert recipe.eval_split == "val"
        assert recipe.max_eval_batches == 6
