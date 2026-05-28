from pathlib import Path

from vlm_diffadapter.dataset_import import load_dataset_import_request


def test_coco8k_dataset_recipe_points_to_expanded_train_pool() -> None:
    request = load_dataset_import_request("configs/datasets/coco2017_train_8k.yaml")

    assert request.dataset_id == "phiyodr/coco2017"
    assert request.split == "train"
    assert request.output_root == Path("data/coco2017_train_8k")
    assert request.image_column == "coco_url"
    assert request.caption_column == "captions"
    assert request.id_column == "image_id"
    assert request.clip_score_column is None
    assert request.limit == 8192
