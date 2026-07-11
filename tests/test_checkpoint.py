"""Checkpoint tests: save/reload, global-step resume, optimizer & scheduler restoration."""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from coordinator_bert.checkpointing import load_checkpoint, save_checkpoint
from coordinator_bert.configuration import TrainConfig
from coordinator_bert.model import BertForMaskedLM

from pretrain_mlm import build_optimizer, build_scheduler  # noqa: E402


def _train_cfg() -> TrainConfig:
    return TrainConfig(warmup_steps=5, max_steps=50, learning_rate=1e-3)


def _step_model(model, optimizer, scheduler, cfg, steps):
    for _ in range(steps):
        ids = torch.randint(5, cfg.vocab_size, (2, 8))
        labels = torch.randint(5, cfg.vocab_size, (2, 8))
        loss = model(ids, labels=labels)["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def test_save_and_reload_weights(tiny_config, tmp_path):
    model = BertForMaskedLM(tiny_config)
    tcfg = _train_cfg()
    opt = build_optimizer(model, tcfg)
    sch = build_scheduler(opt, tcfg)
    _step_model(model, opt, sch, tiny_config, steps=3)

    ckpt = str(tmp_path / "ck")
    save_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sch, global_step=3,
                    config=tiny_config)
    assert os.path.exists(os.path.join(ckpt, "state.pt"))
    assert os.path.exists(os.path.join(ckpt, "meta.json"))

    # Fresh model loads the exact weights.
    model2 = BertForMaskedLM(tiny_config)
    load_checkpoint(ckpt, model=model2, restore_rng=False)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1.data, p2.data), f"mismatch at {n1}"


def test_resume_global_step_and_optimizer(tiny_config, tmp_path):
    model = BertForMaskedLM(tiny_config)
    tcfg = _train_cfg()
    opt = build_optimizer(model, tcfg)
    sch = build_scheduler(opt, tcfg)
    _step_model(model, opt, sch, tiny_config, steps=10)

    ckpt = str(tmp_path / "ck")
    save_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sch, global_step=10,
                    config=tiny_config)

    model2 = BertForMaskedLM(tiny_config)
    opt2 = build_optimizer(model2, tcfg)
    sch2 = build_scheduler(opt2, tcfg)
    payload = load_checkpoint(ckpt, model=model2, optimizer=opt2, scheduler=sch2,
                              restore_rng=False)

    assert payload["global_step"] == 10
    # Scheduler restored to the same LR / step count.
    assert sch2.last_epoch == sch.last_epoch
    torch.testing.assert_close(
        torch.tensor(sch2.get_last_lr()), torch.tensor(sch.get_last_lr())
    )
    # Optimizer state (AdamW moment buffers) restored: step counts match.
    s1 = opt.state_dict()["state"]
    s2 = opt2.state_dict()["state"]
    assert s1.keys() == s2.keys()
    for k in s1:
        assert torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"])


def test_resume_continuation_matches_uninterrupted(tiny_config, tmp_path):
    """Training A->save->resume->B should match training A+B without interruption."""
    torch.manual_seed(1234)
    model = BertForMaskedLM(tiny_config)
    tcfg = _train_cfg()
    opt = build_optimizer(model, tcfg)
    sch = build_scheduler(opt, tcfg)

    # Fixed data so both runs see identical batches.
    batches = [
        (torch.randint(5, tiny_config.vocab_size, (2, 8)),
         torch.randint(5, tiny_config.vocab_size, (2, 8)))
        for _ in range(8)
    ]

    def run(m, o, s, seq):
        for ids, labels in seq:
            loss = m(ids, labels=labels)["loss"]
            loss.backward()
            o.step()
            s.step()
            o.zero_grad()

    run(model, opt, sch, batches[:4])
    ckpt = str(tmp_path / "ck")
    save_checkpoint(ckpt, model=model, optimizer=opt, scheduler=sch, global_step=4,
                    config=tiny_config)
    run(model, opt, sch, batches[4:])
    reference = {n: p.detach().clone() for n, p in model.named_parameters()}

    # Resume path: new objects, load, continue with the same remaining batches.
    model2 = BertForMaskedLM(tiny_config)
    opt2 = build_optimizer(model2, tcfg)
    sch2 = build_scheduler(opt2, tcfg)
    load_checkpoint(ckpt, model=model2, optimizer=opt2, scheduler=sch2, restore_rng=False)
    run(model2, opt2, sch2, batches[4:])

    for n, p in model2.named_parameters():
        torch.testing.assert_close(p.detach(), reference[n], rtol=1e-5, atol=1e-6)
