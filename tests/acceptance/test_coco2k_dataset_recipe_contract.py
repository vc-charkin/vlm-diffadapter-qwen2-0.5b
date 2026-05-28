from pathlib import Path

from vlm_diffadapter.dataset_import import load_dataset_import_request


def test_coco2k_dataset_recipe_points_to_larger_train_pool() -> None:
    request = load_dataset_import_request("configs/datasets/coco2017_trainval_2k.yaml")

    assert request.dataset_id == "phiyodr/coco2017"
    assert request.split == "train"
    assert request.output_root == Path("data/coco2017_trainval_2k")
    assert request.image_column == "coco_url"
    assert request.caption_column == "captions"
    assert request.id_column == "image_id"
    assert request.clip_score_column is None
    assert request.limit == 2048
