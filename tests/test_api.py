"""End-to-end API tests.

These run the real application against a temporary database and upload
directory (see the ``api_environment`` fixture). Nothing is mocked: a
recognition test actually loads a checkpoint and runs the model, so it is
skipped when no model has been trained on this machine.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from ml.utils.image_io import encode_image


def _png_bytes(image: np.ndarray) -> bytes:
    return encode_image(image, extension=".png")


def _upload(image: np.ndarray, name: str = "test.png"):
    return {"file": (name, io.BytesIO(_png_bytes(image)), "image/png")}


def _first_available_model(client) -> str | None:
    payload = client.get("/api/models").json()
    for model in payload["models"]:
        # Only project-trained models are guaranteed usable offline; TrOCR
        # needs a download.
        if model["available"] and model["provenance"] == "trained_in_project":
            return model["id"]
    return None


def test_cnn_transformer_training_job_is_listed(client) -> None:
    payload = client.get("/api/training/jobs/available").json()
    jobs = {job["key"]: job for job in payload["jobs"]}
    assert jobs["cnn_transformer_train"]["ready"] is True
    assert "Transformer" in jobs["cnn_transformer_train"]["label"]


# --------------------------------------------------------------------------
# system
# --------------------------------------------------------------------------
def test_health_reports_subsystem_state(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] in ("ok", "degraded")
    assert payload["database_ready"] is True
    assert payload["device"] in ("cpu", "cuda") or payload["device"].startswith("cuda")


def test_system_info_reports_hardware(client) -> None:
    payload = client.get("/api/system/info").json()
    assert payload["torch_version"]
    assert isinstance(payload["cuda_available"], bool)
    assert isinstance(payload["gpus"], list)


def test_root_redirects_to_the_docs(client) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert "/docs" in response.headers["location"]


def test_openapi_schema_is_served(client) -> None:
    assert client.get("/openapi.json").status_code == 200


def test_every_response_carries_a_request_id(client) -> None:
    response = client.get("/api/health")
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def test_model_list_shape(client) -> None:
    payload = client.get("/api/models").json()
    assert "models" in payload

    for model in payload["models"]:
        assert model["provenance"] in ("trained_in_project", "pretrained_third_party")
        assert model["id"] and model["name"]


def test_model_list_still_validates_after_a_model_is_loaded(client) -> None:
    """Listing models must work once one is in memory.

    A loaded model merges its checkpoint metadata into the response, and a
    checkpoint field called `architecture` (a dict) was overwriting the
    schema's `architecture` (a string), making GET /api/models return 500.
    The earlier list test only ran against an empty registry, so it never saw
    this.
    """
    model_id = _first_available_model(client)
    if model_id is None:
        pytest.skip("no model trained on this machine")

    assert client.post(f"/api/models/{model_id}/load").status_code == 200

    response = client.get("/api/models")
    assert response.status_code == 200, response.text

    loaded = [m for m in response.json()["models"] if m["id"] == model_id]
    assert loaded and loaded[0]["loaded"] is True
    # The typed field must stay a string, whatever the checkpoint carried.
    assert isinstance(loaded[0]["architecture"], str)


def test_single_model_detail_validates_when_loaded(client) -> None:
    model_id = _first_available_model(client)
    if model_id is None:
        pytest.skip("no model trained on this machine")

    client.post(f"/api/models/{model_id}/load")
    response = client.get(f"/api/models/{model_id}")
    assert response.status_code == 200, response.text


def test_unknown_model_is_rejected(client) -> None:
    response = client.get("/api/models/not-a-real-model")
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_unloading_a_model_that_is_not_loaded_is_not_an_error(client) -> None:
    model_id = _first_available_model(client)
    if model_id is None:
        pytest.skip("no model trained on this machine")

    payload = client.post(f"/api/models/{model_id}/unload").json()
    assert "was_loaded" in payload["detail"]


def test_refresh_rescans_the_checkpoint_directory(client) -> None:
    assert client.post("/api/models/refresh").status_code == 200


# --------------------------------------------------------------------------
# upload validation
# --------------------------------------------------------------------------
def test_garbage_upload_is_rejected(client) -> None:
    response = client.post(
        "/api/ocr/recognize",
        files={"file": ("bad.png", io.BytesIO(b"definitely not a png"), "image/png")},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_image"


def test_empty_upload_is_rejected(client) -> None:
    response = client.post(
        "/api/ocr/recognize", files={"file": ("empty.png", io.BytesIO(b""), "image/png")}
    )
    assert response.status_code == 422


def test_unsupported_extension_is_rejected(client) -> None:
    response = client.post(
        "/api/ocr/recognize",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 422
    assert "Accepted formats" in response.json()["message"]


def test_malformed_options_json_is_rejected(client, text_image) -> None:
    response = client.post(
        "/api/ocr/recognize", files=_upload(text_image), data={"options": "{not json"}
    )
    assert response.status_code == 400
    assert "not valid JSON" in response.json()["message"]


def test_invalid_option_value_is_rejected(client, text_image) -> None:
    response = client.post(
        "/api/ocr/recognize",
        files=_upload(text_image),
        data={"options": json.dumps({"beam_width": 999})},
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------
# preprocessing (needs no trained model)
# --------------------------------------------------------------------------
def test_preprocess_preview_returns_a_stage_per_step(client, text_image) -> None:
    response = client.post(
        "/api/ocr/preprocess/preview",
        files=_upload(text_image),
        data={
            "options": json.dumps(
                {"handwritten": False, "preprocess": {"clahe": True, "binarize": "otsu"}}
            ),
            "detect_lines": "true",
        },
    )
    assert response.status_code == 200

    payload = response.json()
    names = [stage["name"] for stage in payload["stages"]]
    # deskew is a default-on step, so it appears without being requested.
    assert names == ["input", "grayscale", "polarity", "deskew", "clahe", "binarize"]
    assert all(stage["image"].startswith("data:image/png;base64,") for stage in payload["stages"])


def test_preprocess_preview_detects_lines(client, multiline_image) -> None:
    response = client.post(
        "/api/ocr/preprocess/preview",
        files=_upload(multiline_image),
        data={"options": json.dumps({"handwritten": False}), "detect_lines": "true"},
    )
    payload = response.json()
    assert len(payload["detected_lines"]) == 4
    assert payload["overlay"].startswith("data:image/png;base64,")


def test_preprocess_preview_defaults_to_handwritten_page_cleanup(client, multiline_image) -> None:
    response = client.post(
        "/api/ocr/preprocess/preview",
        files=_upload(multiline_image),
        data={"detect_lines": "false"},
    )
    assert response.status_code == 200
    assert any(stage["name"] == "handwriting_clean" for stage in response.json()["stages"])


def test_preprocess_preview_is_not_saved_to_history(client, text_image) -> None:
    before = client.get("/api/history").json()["total"]
    client.post("/api/ocr/preprocess/preview", files=_upload(text_image))
    assert client.get("/api/history").json()["total"] == before


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
def test_dataset_listing_includes_the_public_benchmarks(client) -> None:
    payload = client.get("/api/datasets").json()
    keys = {dataset["key"] for dataset in payload["external"]}
    assert {"iam", "iiit5k", "mjsynth", "synthtext"} <= keys

    for dataset in payload["external"]:
        assert dataset["url"].startswith("http")
        assert dataset["preparation"]


def test_demo_images_endpoint(client) -> None:
    payload = client.get("/api/datasets/demo").json()
    assert "images" in payload

    if payload["images"]:
        name = payload["images"][0]["file"]
        response = client.get(f"/api/datasets/demo/{name}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


def test_demo_path_traversal_is_refused(client) -> None:
    response = client.get("/api/datasets/demo/..%2F..%2F.env")
    assert response.status_code == 404


def test_unknown_dataset_is_a_404(client) -> None:
    assert client.get("/api/datasets/not-a-dataset").status_code == 404


# --------------------------------------------------------------------------
# evaluation and charts
# --------------------------------------------------------------------------
def test_empty_evaluation_list(client) -> None:
    payload = client.get("/api/evaluation/runs").json()
    assert payload["total"] == 0 and payload["items"] == []


def test_comparison_explains_itself_when_not_ready(client) -> None:
    payload = client.get("/api/evaluation/comparison").json()
    assert payload["comparable"] is False
    # The summary must say what is missing, not just report false.
    assert len(payload["summary"]) > 40


@pytest.mark.parametrize(
    "chart_id",
    ["model_comparison", "accuracy_vs_speed", "training_curves", "simclr_progress", "history_activity"],
)
def test_charts_return_a_figure_or_an_explained_empty(client, chart_id: str) -> None:
    payload = client.get(f"/api/evaluation/charts/{chart_id}").json()
    assert payload["id"] == chart_id
    if payload["empty"]:
        assert payload["empty_reason"]
    else:
        assert "data" in payload["figure"] and "layout" in payload["figure"]


def test_unknown_chart_is_rejected(client) -> None:
    assert client.get("/api/evaluation/charts/not-a-chart").status_code == 400


def test_missing_training_history_is_a_404(client) -> None:
    assert client.get("/api/evaluation/training-history/never_trained").status_code == 404


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def test_available_jobs_lists_the_ocr_training_options(client) -> None:
    payload = client.get("/api/training/jobs/available").json()
    keys = {job["key"] for job in payload["jobs"]}
    assert keys == {
        "simclr_pretrain",
        "baseline_scratch",
        "simclr_finetune",
        "mamba_train",
        "cnn_transformer_train",
    }

    for job in payload["jobs"]:
        if not job["ready"]:
            assert job["blocked_reason"]


def test_unknown_training_job_is_rejected(client) -> None:
    response = client.post("/api/training/jobs", json={"job": "make_me_a_sandwich"})
    assert response.status_code == 400


def test_training_job_list_is_initially_empty(client) -> None:
    assert client.get("/api/training/jobs").json()["total"] == 0


def test_cancelling_an_unknown_job_is_a_404(client) -> None:
    assert client.post("/api/training/jobs/999/cancel").status_code == 404


# --------------------------------------------------------------------------
# recognition (requires a trained model)
# --------------------------------------------------------------------------
def test_recognition_round_trip(client, text_image) -> None:
    model_id = _first_available_model(client)
    if model_id is None:
        pytest.skip("no model trained on this machine")

    response = client.post(
        "/api/ocr/recognize",
        files=_upload(text_image),
        data={"model_id": model_id, "options": json.dumps({"single_line": True})},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["model_id"] == model_id
    assert payload["id"] is not None
    assert payload["timings_ms"]["total"] > 0
    assert 0 <= payload["confidence"] <= 1
    # A real model on a clean rendered image should produce something.
    assert isinstance(payload["text"], str)


def test_recognition_with_a_ground_truth_is_scored(client, text_image) -> None:
    model_id = _first_available_model(client)
    if model_id is None:
        pytest.skip("no model trained on this machine")

    payload = client.post(
        "/api/ocr/recognize",
        files=_upload(text_image),
        data={"model_id": model_id, "ground_truth": "Hello World 123"},
    ).json()

    assert payload["cer"] is not None and payload["cer"] >= 0
    assert payload["wer"] is not None
    assert isinstance(payload["exact_match"], bool)


def test_history_records_a_recognition(client, text_image) -> None:
    model_id = _first_available_model(client)
    if model_id is None:
        pytest.skip("no model trained on this machine")

    before = client.get("/api/history").json()["total"]
    created = client.post(
        "/api/ocr/recognize", files=_upload(text_image), data={"model_id": model_id}
    ).json()

    listing = client.get("/api/history").json()
    assert listing["total"] == before + 1

    detail = client.get(f"/api/history/{created['id']}").json()
    assert detail["id"] == created["id"]
    assert detail["text"] == created["text"]

    # The stored image is retrievable.
    assert client.get(f"/api/history/{created['id']}/image").status_code == 200


def test_ground_truth_can_be_set_and_cleared(client, text_image) -> None:
    model_id = _first_available_model(client)
    if model_id is None:
        pytest.skip("no model trained on this machine")

    created = client.post(
        "/api/ocr/recognize", files=_upload(text_image), data={"model_id": model_id}
    ).json()

    scored = client.put(
        f"/api/history/{created['id']}/ground-truth", json={"ground_truth": "Hello World 123"}
    ).json()
    assert scored["cer"] is not None

    cleared = client.put(
        f"/api/history/{created['id']}/ground-truth", json={"ground_truth": ""}
    ).json()
    assert cleared["cer"] is None and cleared["ground_truth"] is None


def test_history_entry_can_be_deleted(client, text_image) -> None:
    model_id = _first_available_model(client)
    if model_id is None:
        pytest.skip("no model trained on this machine")

    created = client.post(
        "/api/ocr/recognize", files=_upload(text_image), data={"model_id": model_id}
    ).json()

    assert client.delete(f"/api/history/{created['id']}").status_code == 200
    assert client.get(f"/api/history/{created['id']}").status_code == 404


def test_clearing_history_requires_confirmation(client) -> None:
    response = client.delete("/api/history")
    assert response.status_code == 400
    assert "confirm=true" in response.json()["message"]


def test_save_to_history_can_be_disabled(client, text_image) -> None:
    model_id = _first_available_model(client)
    if model_id is None:
        pytest.skip("no model trained on this machine")

    before = client.get("/api/history").json()["total"]
    payload = client.post(
        "/api/ocr/recognize",
        files=_upload(text_image),
        data={"model_id": model_id, "options": json.dumps({"save_to_history": False})},
    ).json()

    assert payload["id"] is None
    assert client.get("/api/history").json()["total"] == before


def test_comparison_runs_every_selected_model(client, text_image) -> None:
    model_id = _first_available_model(client)
    if model_id is None:
        pytest.skip("no model trained on this machine")

    payload = client.post(
        "/api/ocr/compare",
        files=_upload(text_image),
        data={"model_ids": model_id, "ground_truth": "Hello World 123"},
    ).json()

    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["model_id"] == model_id
    assert result["cer"] is not None


def test_comparison_requires_at_least_one_model(client, text_image) -> None:
    # A field that is absent or empty is caught by request validation (422);
    # one that is present but names no model reaches the handler (400). Both
    # must be refused with a message that says what is wrong.
    missing = client.post("/api/ocr/compare", files=_upload(text_image), data={"model_ids": ""})
    assert missing.status_code in (400, 422)
    assert missing.json()["message"]

    blank = client.post("/api/ocr/compare", files=_upload(text_image), data={"model_ids": " , "})
    assert blank.status_code == 400
    assert "at least one model" in blank.json()["message"].lower()


def test_comparison_rejects_an_unknown_model(client, text_image) -> None:
    response = client.post(
        "/api/ocr/compare", files=_upload(text_image), data={"model_ids": "imaginary_model"}
    )
    assert response.status_code == 400
    assert "imaginary_model" in response.json()["message"]


def test_history_stats_reflect_activity(client, text_image) -> None:
    model_id = _first_available_model(client)
    if model_id is None:
        pytest.skip("no model trained on this machine")

    client.post("/api/ocr/recognize", files=_upload(text_image), data={"model_id": model_id})
    stats = client.get("/api/history/stats").json()

    assert stats["total_requests"] >= 1
    assert model_id in stats["requests_by_model"]


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def test_export_txt_returns_the_text_verbatim(client) -> None:
    text = "Invoice No: INV-44718\nTax at 18% $1,250.00\n\nNew paragraph."
    response = client.post("/api/ocr/export/txt", json={"text": text, "filename": "demo"})

    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert "demo.txt" in response.headers["content-disposition"]
    assert response.content.decode("utf-8-sig").replace("\r\n", "\n") == text


def test_export_docx_returns_a_word_document(client) -> None:
    response = client.post(
        "/api/ocr/export/docx",
        json={"text": "Line one\nLine two", "filename": "demo", "title": "Extracted text"},
    )
    assert response.status_code == 200
    assert "demo.docx" in response.headers["content-disposition"]
    # A .docx is a zip; its magic bytes are PK.
    assert response.content[:2] == b"PK"


def test_export_rejects_a_missing_text_field(client) -> None:
    assert client.post("/api/ocr/export/txt", json={"filename": "x"}).status_code == 422


def test_export_filename_cannot_escape_the_download(client) -> None:
    response = client.post(
        "/api/ocr/export/txt", json={"text": "x", "filename": "../../etc/passwd"}
    )
    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert ".." not in disposition and "/" not in disposition.split("filename=")[1]


def test_preprocess_preview_reports_perspective(client, text_image) -> None:
    response = client.post(
        "/api/ocr/preprocess/preview",
        files=_upload(text_image),
        data={"options": json.dumps({"preprocess": {"perspective": True}})},
    )
    assert response.status_code == 200

    payload = response.json()
    assert "perspective" in [stage["name"] for stage in payload["stages"]]
    # No page outline in a plain line crop, so it must report doing nothing.
    assert payload["metadata"]["perspective_corrected"] is False
