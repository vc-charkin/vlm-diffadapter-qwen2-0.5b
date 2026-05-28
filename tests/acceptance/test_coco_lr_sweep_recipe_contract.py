from pathlib import Path

from vlm_diffadapter.config import load_train_config
from vlm_diffadapter.experiments import load_experiment_recipe


def test_coco_h100_seed46_lr_sweep_configs_are_versioned() -> None:
    expected_train_configs = {
        "configs/train_image_lr_5e_5.yaml": 0.00005,
        "configs/train_image_lr_2e_4.yaml": 0.0002,
    }

    for train_config_path, image_lr in expected_train_configs.items():
        train_config = load_train_config(train_config_path)
        assert train_config.batch_size == 2
        assert train_config.learning_rates.text == 0.0
        assert train_config.learning_rates.image == image_lr
        assert train_config.loss_weights.lm == 1.0
        assert train_config.loss_weights.diffusion == 1.0

    expected_recipes = {
        "configs/experiments/coco_h100_datamodule_16batch_seed46_lr5e_5.yaml": (
            "h100_real_coco_datamodule_16batch_seed46_lr5e_5",
            Path("configs/train_image_lr_5e_5.yaml"),
        ),
        "configs/experiments/coco_h100_datamodule_16batch_seed46_lr2e_4.yaml": (
            "h100_real_coco_datamodule_16batch_seed46_lr2e_4",
            Path("configs/train_image_lr_2e_4.yaml"),
        ),
    }

    for recipe_path, (run_name, train_config_path) in expected_recipes.items():
        recipe = load_experiment_recipe(recipe_path)
        assert recipe.run_name == run_name
        assert recipe.output_root == Path("runs")
        assert recipe.model_config == Path("configs/model_h100_real.yaml")
        assert recipe.train_config == train_config_path
        assert recipe.seed == 46
        assert recipe.adapter_only_checkpoint is True
        assert recipe.device == "cuda"
        assert recipe.manifest == Path("data/coco2017_smoke_128/manifest.jsonl")
        assert recipe.text_length == 8
        assert recipe.use_data_module is True
        assert recipe.val_fraction == 0.1
        assert recipe.max_train_batches == 16
        assert recipe.eval_split == "val"
        assert recipe.max_eval_batches == 6
