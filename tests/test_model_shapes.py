"""Model tests: forward shapes, MLM logits, tied embeddings, gradients, param count."""

from __future__ import annotations

from dataclasses import replace

import torch

from coordinator_bert.model import (
    BertForMaskedLM,
    BertModel,
    count_parameters,
)


def test_encoder_forward_shapes(tiny_config):
    model = BertModel(tiny_config).eval()
    ids = torch.randint(5, tiny_config.vocab_size, (3, 9))
    out = model(ids, attention_mask=torch.ones(3, 9, dtype=torch.long))
    assert out["last_hidden_state"].shape == (3, 9, tiny_config.hidden_size)
    assert out["pooler_output"].shape == (3, tiny_config.hidden_size)


def test_mlm_logits_shape(tiny_config):
    model = BertForMaskedLM(tiny_config).eval()
    ids = torch.randint(5, tiny_config.vocab_size, (2, 11))
    out = model(ids)
    assert out["logits"].shape == (2, 11, tiny_config.vocab_size)


def test_mlm_loss_computed_with_labels(tiny_config):
    model = BertForMaskedLM(tiny_config)
    ids = torch.randint(5, tiny_config.vocab_size, (2, 11))
    labels = torch.full((2, 11), -100)
    labels[:, 3] = torch.randint(5, tiny_config.vocab_size, (2,))
    out = model(ids, labels=labels)
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])


def test_tied_embeddings_share_storage(tiny_config):
    model = BertForMaskedLM(tiny_config)
    w_in = model.bert.embeddings.word_embeddings.weight
    w_out = model.mlm_head.decoder.weight
    assert w_in.data_ptr() == w_out.data_ptr(), "input/output embeddings are not tied"
    # A grad on the output decoder must appear on the input embedding (same tensor).
    model.zero_grad()
    ids = torch.randint(5, tiny_config.vocab_size, (2, 8))
    labels = torch.randint(5, tiny_config.vocab_size, (2, 8))
    model(ids, labels=labels)["loss"].backward()
    assert w_in.grad is not None


def test_untied_embeddings_are_separate():
    from coordinator_bert.configuration import ModelConfig

    cfg = ModelConfig(
        vocab_size=128, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=32, tie_word_embeddings=False,
    )
    model = BertForMaskedLM(cfg)
    assert (
        model.bert.embeddings.word_embeddings.weight.data_ptr()
        != model.mlm_head.decoder.weight.data_ptr()
    )


def test_no_nan_in_forward(tiny_config):
    model = BertForMaskedLM(tiny_config).eval()
    ids = torch.randint(5, tiny_config.vocab_size, (4, 16))
    out = model(ids, attention_mask=torch.ones(4, 16, dtype=torch.long))
    assert torch.isfinite(out["logits"]).all()


def test_gradient_propagation(tiny_config):
    model = BertForMaskedLM(tiny_config)
    model.zero_grad()
    ids = torch.randint(5, tiny_config.vocab_size, (2, 12))
    labels = torch.randint(5, tiny_config.vocab_size, (2, 12))
    model(ids, labels=labels)["loss"].backward()

    # The pooler is intentionally NOT part of the MLM loss path, so it legitimately gets no
    # gradient during MLM pretraining. Every OTHER trainable parameter must receive one.
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None and not n.startswith("bert.pooler")]
    assert not missing, f"parameters without gradient: {missing}"
    # At least one gradient is non-zero.
    total = sum(float(p.grad.abs().sum()) for _, p in model.named_parameters()
                if p.grad is not None)
    assert total > 0


def test_pooler_not_in_mlm_loss_path(tiny_config):
    """Documents that the pooler does not participate in the MLM objective."""
    model = BertForMaskedLM(tiny_config)
    model.zero_grad()
    ids = torch.randint(5, tiny_config.vocab_size, (2, 8))
    labels = torch.randint(5, tiny_config.vocab_size, (2, 8))
    model(ids, labels=labels)["loss"].backward()
    assert model.bert.pooler.dense.weight.grad is None


def test_pre_norm_variant_runs(tiny_config):
    cfg = replace(tiny_config, norm_type="pre")
    model = BertForMaskedLM(cfg).eval()
    ids = torch.randint(5, cfg.vocab_size, (2, 10))
    out = model(ids)
    assert out["logits"].shape == (2, 10, cfg.vocab_size)
    assert torch.isfinite(out["logits"]).all()
    assert model.bert.encoder.final_norm is not None


def test_parameter_count_matches_estimate(tiny_config):
    model = BertForMaskedLM(tiny_config)
    counts = count_parameters(model)
    estimate = tiny_config.estimate_num_parameters()
    # Closed-form estimate should match the real (deduplicated) count exactly.
    assert counts["unique"] == estimate, (counts["unique"], estimate)


def test_token_type_disabled_config():
    from coordinator_bert.configuration import ModelConfig

    cfg = ModelConfig(
        vocab_size=128, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=32, type_vocab_size=0,
    )
    model = BertForMaskedLM(cfg).eval()
    assert model.bert.embeddings.token_type_embeddings is None
    ids = torch.randint(5, cfg.vocab_size, (2, 8))
    assert model(ids)["logits"].shape == (2, 8, cfg.vocab_size)
