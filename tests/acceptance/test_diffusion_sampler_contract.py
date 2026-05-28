from vlm_diffadapter.config import load_model_config
from vlm_diffadapter.inference import generate_image
from vlm_diffadapter.modeling import VlmDiffAdapter


def test_generate_image_uses_model_sampler_instead_of_solid_placeholder() -> None:
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))

    first = generate_image(
        model,
        "A red robot in a library",
        generation_config={"num_inference_steps": 4},
        seed=123,
        size=(64, 64),
    )
    repeated = generate_image(
        model,
        "A red robot in a library",
        generation_config={"num_inference_steps": 4},
        seed=123,
        size=(64, 64),
    )
    other_prompt = generate_image(
        model,
        "A blue train in a forest",
        generation_config={"num_inference_steps": 4},
        seed=123,
        size=(64, 64),
    )

    assert first.size == (64, 64)
    assert first.tobytes() == repeated.tobytes()
    assert len(first.getcolors(maxcolors=64 * 64)) > 1
    assert first.tobytes() != other_prompt.tobytes()


def test_generate_image_supports_classifier_free_guidance() -> None:
    model = VlmDiffAdapter(load_model_config("configs/model.yaml"))

    unguided = generate_image(
        model,
        "A red robot in a library",
        generation_config={"num_inference_steps": 4, "guidance_scale": 1.0},
        seed=123,
        size=(64, 64),
    )
    guided = generate_image(
        model,
        "A red robot in a library",
        generation_config={"num_inference_steps": 4, "guidance_scale": 3.0},
        seed=123,
        size=(64, 64),
    )

    assert guided.size == (64, 64)
    assert guided.tobytes() != unguided.tobytes()
