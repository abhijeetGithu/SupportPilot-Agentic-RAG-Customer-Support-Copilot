"""The production classifier path (int8 ONNX on onnxruntime), with a tiny generated model and tokenizer."""

import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from supportpilot.classifier.onnx_model import OnnxClassifier


def tiny_model(folder, logits_per_token: np.ndarray, labels: list[str], temperature: float) -> None:
    """logits = mean over tokens of a per-token logit table (embedding lookup), masked by attention."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "billing": 1, "crawl": 2, "error": 3}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(folder / "tokenizer.json"))
    table = helper.make_tensor("table", TensorProto.FLOAT, logits_per_token.shape, logits_per_token.flatten())
    nodes = [
        helper.make_node("Gather", ["table", "input_ids"], ["per_token"]),           # [1, seq, labels]
        helper.make_node("ReduceMean", ["per_token"], ["logits"], axes=[1], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, None]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, None])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, len(labels)])],
        initializer=[table],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(folder / "model.int8.onnx"))
    (folder / "meta.json").write_text(json.dumps({"labels": labels, "temperature": temperature, "max_length": 16}))


def test_onnx_classifier_predicts_with_temperature(tmp_path):
    table = np.array([[0, 0], [4, 0], [0, 3], [0, 1]], dtype=np.float32)  # token -> logits for [Billing, Technical]
    tiny_model(tmp_path, table, ["Billing", "Technical"], temperature=2.0)
    clf = OnnxClassifier(tmp_path)
    label, conf = clf.predict("billing")
    assert label == "Billing" and conf == pytest.approx(1 / (1 + np.exp(-2.0)), rel=1e-5)  # softmax(4/2, 0)
    label, conf = clf.predict("crawl error")
    assert label == "Technical" and conf == pytest.approx(1 / (1 + np.exp(-1.0)), rel=1e-5)  # mean (0,2) / 2


def test_predict_prefers_onnx_and_can_release_models(tmp_path, monkeypatch):
    from supportpilot.classifier import predict
    from supportpilot.config import get_settings

    for task, labels in (("queue", ["Billing", "Technical"]), ("priority", ["low", "high"])):
        (tmp_path / task).mkdir()
        tiny_model(tmp_path / task, np.array([[0, 0], [4, 0], [0, 3], [0, 1]], dtype=np.float32), labels, 1.0)
    monkeypatch.setenv("CLASSIFIER_ONNX_DIR", str(tmp_path))
    monkeypatch.setenv("CLASSIFIER_KEEP_LOADED", "false")
    get_settings.cache_clear()
    predict._cached_task.cache_clear()
    try:
        p = predict.classify_ticket("crawl error")
        assert (p.category, p.priority, p.model) == ("Technical", "high", "distilbert-int8")
        assert predict._cached_task.cache_info().currsize == 0  # nothing kept in memory
    finally:
        get_settings.cache_clear()
        predict._cached_task.cache_clear()
