# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib.util
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.parts.utils.asr_batching import (
    COST_MODEL_WEIGHTS,
    DurationCostBatchSampler,
    SemiSortBatchSampler,
    get_duration_cost_batch_sampler,
    rnnt_batch_cost,
)
from nemo.collections.asr.parts.utils.manifest_utils import write_manifest

#: The fine-tune entry point that owns the validation rebuild. It is an
#: ``examples/`` script, not an importable package, so it is loaded by path.
_FINETUNE_SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "asr" / "speech_to_text_finetune_cached.py"

_finetune_module = None


def _load_finetune_module():
    """Import the fine-tune script by path, undoing its global ``torch.load`` patch.

    The script sets ``torch.load``'s default to ``weights_only=False`` for the
    whole process at import time (it has to, to read NeMo checkpoints). Here that
    would leak into every other test in the run, so it is put back.
    """
    global _finetune_module
    if _finetune_module is None:
        saved_load = torch.load
        spec = importlib.util.spec_from_file_location("finetune_cached_under_test", _FINETUNE_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        finally:
            torch.load = saved_load
        _finetune_module = module
    return _finetune_module


class TestASRSamplers:
    labels = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ]

    @pytest.mark.unit
    def test_ssb_sampler(self):
        # Generate random signals
        data_min_duration = 0.1
        data_max_duration = 16.7

        random_seed = 42
        sample_rate = 16000

        _rng = np.random.default_rng(seed=random_seed)

        def generate_samples(num_examples: int) -> list:
            data_duration = np.round(_rng.uniform(low=data_min_duration, high=data_max_duration, size=num_examples), 3)
            data_duration_samples = np.floor(data_duration * sample_rate).astype(int)
            samples = []
            for data_duration_sample in data_duration_samples:
                samples.append(_rng.uniform(low=-0.5, high=0.5, size=(data_duration_sample)))
            return samples

        with tempfile.TemporaryDirectory() as test_dir:
            # Build metadata for manifest
            metadata = []

            # Test size of dataloader with and without ssb
            for num_samples in np.concatenate([np.array([1, 2]), _rng.integers(3, 10, 2), _rng.integers(10, 1000, 2)]):
                samples = generate_samples(num_samples)

                for n, sample in enumerate(samples):
                    meta = dict()
                    signal_filename = f'{n:04d}.wav'
                    # write audio files
                    sf.write(os.path.join(test_dir, signal_filename), sample, sample_rate)
                    # update metadata
                    meta['audio_filepath'] = os.path.join(test_dir, signal_filename)
                    meta['duration'] = len(sample) / sample_rate
                    meta['text'] = 'non empty'
                    metadata.append(meta)

                # Save manifest
                manifest_filepath = os.path.join(test_dir, 'manifest.json')
                write_manifest(manifest_filepath, metadata)

                # Make dataset
                dataset = audio_to_text.AudioToCharDataset(
                    manifest_filepath=manifest_filepath,
                    labels=self.labels,
                    sample_rate=sample_rate,
                    max_duration=data_max_duration,
                    min_duration=data_min_duration,
                )
                durations = [sample.duration for sample in dataset.manifest_processor.collection.data]

                # Compare two dataloader
                for batch_size in _rng.integers(1, n + 20, 5):
                    batch_size = int(batch_size)
                    drop_last = True if _rng.integers(0, 2) else False
                    sampler = SemiSortBatchSampler(
                        global_rank=0,
                        world_size=1,
                        durations=durations,
                        batch_size=batch_size,
                        batch_shuffle=True,
                        drop_last=drop_last,
                        randomization_factor=0.1,
                        seed=random_seed,
                    )
                    dataloader_with_ssb = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=None,
                        sampler=sampler,
                        batch_sampler=None,
                        collate_fn=lambda x: audio_to_text._speech_collate_fn(x, pad_id=0),
                    )
                    dataloader = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=batch_size,
                        collate_fn=lambda x: audio_to_text._speech_collate_fn(x, pad_id=0),
                        drop_last=drop_last,
                        shuffle=True,
                    )

                    assert abs(len(dataloader) - len(dataloader_with_ssb)) == 0, (
                        "Different num of batches with batch! Num of batches with ssb is "
                        f"{len(dataloader_with_ssb)} and without ssb is {len(dataloader)}!"
                    )

                    dataloader_with_ssb_exception, dataloader_exception = False, False

                    try:
                        list(dataloader_with_ssb)
                    except:
                        dataloader_with_ssb_exception = True

                    try:
                        list(dataloader)
                    except:
                        dataloader_exception = True

                    assert dataloader_with_ssb_exception == dataloader_exception

    # ---- DurationCostBatchSampler (adaptive batch size) ----------------------
    # Added for the UltiMed fine-tune with Claude Code. The sampler decides how
    # much fits on the GPU, so a silent regression here is an OOM ten hours into
    # a multi-day run: these lock down the cost model and every packing bound.

    @staticmethod
    def _durations(seed: int = 0, n: int = 2000) -> np.ndarray:
        """Duration spread close to the real corpus: median ~17 s, tail to 38 s."""
        rng = np.random.default_rng(seed)
        return np.clip(rng.gamma(shape=4.0, scale=4.5, size=n), 1.0, 38.0)

    def test_duration_cost_matches_calibration(self):
        # The budget was fitted against measured VRAM on an RTX 3090 Ti. If this
        # drifts, every budget in training_config.yaml is silently wrong.
        for durations, expected in (
            ([25.0] * 4, 13750.0),
            ([40.0], 16000.0),
            ([25.0] * 6, 18750.0),
            ([30.0] * 4, 19800.0),
        ):
            assert rnnt_batch_cost(durations) == pytest.approx(expected)
        assert rnnt_batch_cost([]) == 0.0

    def test_duration_cost_sampler_respects_every_bound(self):
        durations = self._durations()
        budget, max_bs, max_total = 15000.0, 16, 240.0
        sampler = DurationCostBatchSampler(
            durations=durations,
            budget=budget,
            max_batch_size=max_bs,
            max_total_duration=max_total,
            sort_pool_size=512,
        )
        for batch in sampler:
            assert len(batch) <= max_bs
            assert durations[batch].sum() <= max_total
            # a single clip has no smaller batch to fall back to, so it is
            # allowed through even if it busts the budget
            if len(batch) > 1:
                assert rnnt_batch_cost(durations[batch]) <= budget

    def test_duration_cost_sampler_covers_every_sample_once(self):
        durations = self._durations(n=500)
        sampler = DurationCostBatchSampler(durations=durations, budget=15000.0)
        for epoch in range(3):
            sampler.set_epoch(epoch)
            drawn = [i for batch in sampler for i in batch]
            assert sorted(drawn) == list(range(len(durations)))
            # Lightning sizes the progress bar and val_check_interval from
            # __len__ before iterating, so the two must agree.
            assert len(sampler) == len(list(iter(sampler)))

    def test_duration_cost_sampler_reshuffles_but_is_reproducible(self):
        durations = self._durations(n=500)
        first = DurationCostBatchSampler(durations=durations, budget=15000.0, seed=7)
        epoch0 = [list(b) for b in first]
        first.set_epoch(1)
        epoch1 = [list(b) for b in first]
        assert epoch0 != epoch1, "batches must change between epochs"

        same_seed = DurationCostBatchSampler(durations=durations, budget=15000.0, seed=7)
        assert [list(b) for b in same_seed] == epoch0, "same seed must replay the same epoch"

        no_shuffle = DurationCostBatchSampler(durations=durations, budget=15000.0, shuffle=False)
        before = [list(b) for b in no_shuffle]
        no_shuffle.set_epoch(1)
        assert [list(b) for b in no_shuffle] == before, "shuffle=False must be stable"

    def test_duration_cost_sampler_adapts_size_to_duration(self):
        # The whole point: long clips travel alone, short ones in full batches.
        long_only = DurationCostBatchSampler(durations=[38.0] * 20, budget=15000.0)
        assert {len(b) for b in long_only} == {1}

        short_only = DurationCostBatchSampler(
            durations=[5.0] * 100, budget=15000.0, max_batch_size=16, max_total_duration=240.0
        )
        assert max(len(b) for b in short_only) == 16

    def test_duration_cost_sampler_emits_oversized_clip_alone(self):
        # 60 s alone costs 36000, well over the budget. It must still come out,
        # on its own, rather than be dropped or padded into a bigger batch.
        durations = [60.0, 5.0, 5.0]
        sampler = DurationCostBatchSampler(durations=durations, budget=15000.0, shuffle=False)
        batches = [list(b) for b in sampler]
        assert [0] in batches
        assert sorted(i for b in batches for i in b) == [0, 1, 2]

    def test_duration_cost_sampler_shards_across_ranks(self):
        durations = self._durations(n=500)
        world_size = 3
        shards = [
            DurationCostBatchSampler(durations=durations, budget=15000.0, global_rank=rank, world_size=world_size)
            for rank in range(world_size)
        ]
        assert len({len(s) for s in shards}) == 1, "ranks must agree on the step count or DDP hangs"
        seen = [tuple(b) for s in shards for b in s]
        assert len(seen) == len(set(seen)), "no batch may be handed to two ranks"

    def test_duration_cost_sampler_rejects_bad_config(self):
        with pytest.raises(ValueError):
            DurationCostBatchSampler(durations=[], budget=15000.0)
        with pytest.raises(ValueError):
            DurationCostBatchSampler(durations=[10.0], budget=0)
        with pytest.raises(ValueError):
            DurationCostBatchSampler(durations=[10.0], budget=15000.0, max_batch_size=0)

    # ---- label-length-aware cost model ---------------------------------------
    # The lattice is [T, U, V], so d^2 is only right while U tracks T. Measured
    # 2026-10-28 on v2 (oom_margin_check.sh LONE=40): two lone 39.8 s clips, 264
    # and 319 tokens, peaked 18058 and 22194 MiB at the same d^2 cost.

    def test_duration_sq_stays_the_default(self):
        # Nothing may change silently until the duration_tokens weights are
        # refit on the GPU: a sampler built the old way must still be the old one.
        durations = self._durations(n=500)
        default = DurationCostBatchSampler(durations=durations, budget=15000.0, seed=3)
        explicit = DurationCostBatchSampler(
            durations=durations, budget=15000.0, seed=3, cost_model='duration_sq', sum_weight=4.0, max_weight=6.0
        )
        assert default.cost_model == 'duration_sq'
        assert (default.sum_weight, default.max_weight) == (4.0, 6.0)
        assert [list(b) for b in default] == [list(b) for b in explicit]

    def test_duration_tokens_cost_prices_label_length(self):
        durations = [39.8]
        sparse, dense = [264], [319]
        assert rnnt_batch_cost(durations) == rnnt_batch_cost(durations), "d^2 cannot tell them apart"
        cost_sparse = rnnt_batch_cost(durations, token_counts=sparse, cost_model='duration_tokens')
        cost_dense = rnnt_batch_cost(durations, token_counts=dense, cost_model='duration_tokens')
        sum_w, max_w = COST_MODEL_WEIGHTS['duration_tokens']
        assert cost_sparse == pytest.approx((sum_w + max_w) * 39.8 * 265)
        assert cost_dense / cost_sparse == pytest.approx(320 / 265)
        # Explicit weights still win over the model's defaults.
        assert rnnt_batch_cost(
            [10.0, 20.0], 1.0, 0.0, token_counts=[9, 19], cost_model='duration_tokens'
        ) == pytest.approx(10 * 10 + 20 * 20)

    def test_duration_tokens_sampler_respects_budget_and_sends_dense_clips_alone(self):
        rng = np.random.default_rng(1)
        durations = self._durations(n=2000)
        # 4 to 8 tokens/s, wider than the nine v2 voices' 5.1 to 6.4 means.
        tokens = np.round(durations * rng.uniform(4.0, 8.0, size=durations.size)).astype(int)
        budget = 15000.0
        sampler = DurationCostBatchSampler(
            durations=durations, budget=budget, token_counts=tokens, cost_model='duration_tokens'
        )
        drawn = sorted(i for b in sampler for i in b)
        assert drawn == list(range(len(durations)))
        for batch in sampler:
            if len(batch) > 1:
                assert (
                    rnnt_batch_cost(durations[batch], token_counts=tokens[batch], cost_model='duration_tokens')
                    <= budget
                )
        assert sampler.stats()['cost_max'] <= max(
            budget, max(rnnt_batch_cost([d], token_counts=[u], cost_model='duration_tokens') for d, u in zip(durations, tokens))
        )

        # Same durations, different label lengths: two sparse 30 s clips fit one
        # batch, two dense ones do not. duration_sq would treat both pairs alike.
        sum_w, max_w = COST_MODEL_WEIGHTS['duration_tokens']
        pair_budget = sum_w * 2 * 30 * 151 + max_w * 30 * 151 + 1.0  # two clips of 150 tokens, just fits
        sparse = DurationCostBatchSampler(
            durations=[30.0, 30.0], budget=pair_budget, token_counts=[150, 150],
            cost_model='duration_tokens', shuffle=False,
        )
        dense = DurationCostBatchSampler(
            durations=[30.0, 30.0], budget=pair_budget, token_counts=[240, 240],
            cost_model='duration_tokens', shuffle=False,
        )
        assert [len(b) for b in sparse] == [2]
        assert [len(b) for b in dense] == [1, 1]

    def test_duration_tokens_rejects_bad_config(self):
        with pytest.raises(ValueError, match="token_counts"):
            DurationCostBatchSampler(durations=[10.0], budget=15000.0, cost_model='duration_tokens')
        with pytest.raises(ValueError, match="one count per clip"):
            DurationCostBatchSampler(
                durations=[10.0, 5.0], budget=15000.0, token_counts=[3], cost_model='duration_tokens'
            )
        with pytest.raises(ValueError, match="cost_model"):
            DurationCostBatchSampler(durations=[10.0], budget=15000.0, cost_model='frames')
        with pytest.raises(ValueError, match="cost_model"):
            rnnt_batch_cost([10.0], cost_model='frames')

    @pytest.mark.unit
    def test_duration_tokens_reads_label_lengths_from_the_collection(self):
        # The collection already tokenized every transcript when the manifest
        # loaded; the sampler must reuse that, not tokenize again.
        texts = ['ab', 'abc def', 'a']
        with tempfile.TemporaryDirectory() as test_dir:
            dataset = self._write_dataset(test_dir, 'tok', [1.0, 2.0, 3.0], texts=texts)
            model = SimpleNamespace(global_rank=0, world_size=1)
            on = get_duration_cost_batch_sampler(
                model, dataset, {'cost_batching': {'budget': 100.0, 'cost_model': 'duration_tokens'}}
            )
            expected = [len(sample.text_tokens) for sample in dataset.manifest_processor.collection.data]
            assert expected == [len(t) for t in texts], "char parser: one token per character"
            assert on.cost_model == 'duration_tokens'
            assert list(on.token_counts) == expected
            assert (on.sum_weight, on.max_weight) == COST_MODEL_WEIGHTS['duration_tokens']

            off = get_duration_cost_batch_sampler(model, dataset, {'cost_batching': {'budget': 100.0}})
            assert off.cost_model == 'duration_sq' and off.token_counts is None

            weighted = get_duration_cost_batch_sampler(
                model, dataset, {'cost_batching': {'budget': 100.0, 'sum_weight': 1.5, 'max_weight': 2.5}}
            )
            assert (weighted.sum_weight, weighted.max_weight) == (1.5, 2.5)

    # ---- validation cost batching -------------------------------------------
    # Validation is batched by the fine-tune script, not by the sampler itself,
    # and it is the wiring that fails quietly: one `cost_batching` block has to
    # reach fifteen `ds_item` entries, and if it reaches none of them every set
    # silently stays on its fallback `batch_size` and the only symptom is a slow
    # validation epoch.

    def _write_dataset(self, test_dir, name, durations, sample_rate=16000, texts=None):
        """A real ``AudioToCharDataset`` over clips of the given durations.

        The audio is silence: nothing here listens to it, but the sampler reads
        the manifest durations and the collate function reads the waveforms.
        """
        metadata = []
        texts = texts or ['non empty'] * len(durations)
        for n, (duration, text) in enumerate(zip(durations, texts)):
            filename = os.path.join(test_dir, f'{name}_{n:04d}.wav')
            sf.write(filename, np.zeros(int(duration * sample_rate), dtype='float32'), sample_rate)
            metadata.append({'audio_filepath': filename, 'duration': duration, 'text': text})
        manifest_filepath = os.path.join(test_dir, f'{name}.json')
        write_manifest(manifest_filepath, metadata)
        return audio_to_text.AudioToCharDataset(
            manifest_filepath=manifest_filepath, labels=self.labels, sample_rate=sample_rate
        )

    @staticmethod
    def _collate(batch):
        return audio_to_text._speech_collate_fn(batch, pad_id=0)

    @pytest.mark.unit
    def test_validation_cost_batching_rebuilds_every_dataloader(self):
        finetune = _load_finetune_module()
        with tempfile.TemporaryDirectory() as test_dir:
            mixed = self._write_dataset(test_dir, 'parhaf', [0.5] * 16 + [2.0, 2.0])
            short = self._write_dataset(test_dir, 'fleurs', [1.0] * 8)
            loaders = [
                torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=self._collate, shuffle=False)
                for ds in (mixed, short)
            ]
            model = SimpleNamespace(
                global_rank=0, world_size=1, _validation_dl=loaders,
                _validation_names=['ultimed_parhaf_', 'fleurs_'],
            )
            # The block is written ONCE, at the validation_ds level: no ds_item
            # carries a copy of it, which is the thing under test.
            val_cfg = OmegaConf.create({
                'batch_size': 2,
                'shuffle': False,
                'cost_batching': {'enabled': True, 'budget': 40.0, 'max_batch_size': 16,
                                  'max_total_duration': 30.0, 'sort_pool_size': 512, 'seed': 42},
                'ds_item': [
                    {'name': 'ultimed_parhaf', 'batch_size': 2, 'shuffle': False, 'num_workers': 0},
                    {'name': 'fleurs', 'batch_size': 6, 'shuffle': False, 'num_workers': 0},
                ],
            })

            finetune.apply_cost_batch_sampler_to_validation(model, val_cfg)

            rebuilt = model._validation_dl
            assert len(rebuilt) == 2, "every validation set must be rebuilt"
            for before, after in zip(loaders, rebuilt):
                assert after is not before
                assert isinstance(after.batch_sampler, DurationCostBatchSampler)
                assert after.batch_sampler.budget == 40.0, "the shared block must reach every set"
                assert after.batch_sampler.shuffle is False, "validation order must stay fixed"
                assert after.collate_fn is before.collate_fn, "the NeMo collate must survive"
                assert after.dataset is before.dataset, "the dataset must not be rebuilt"

            # Adaptive where a fixed batch_size of 2 could not be: the sixteen
            # 0.5 s clips travel together (cost 4*16*0.25 + 6*0.25 = 17.5 of the
            # 40 budget) while one 2.0 s clip already costs 4*4 + 6*4 = 40 and so
            # goes alone.
            batches = [list(b) for b in rebuilt[0].batch_sampler]
            assert sorted(i for b in batches for i in b) == list(range(18)), "every clip once"
            assert max(len(b) for b in batches) == 16
            assert all(len(b) == 1 for b in batches if {16, 17} & set(b))

            # A rebuilt loader must still actually yield: shape[0] is the number
            # of clips the sampler put in that batch, not the config batch_size.
            signals = next(iter(rebuilt[1]))[0]
            assert signals.shape[0] == len(list(rebuilt[1].batch_sampler)[0]) == 8

    @pytest.mark.unit
    def test_validation_cost_batching_respects_the_switch_and_the_single_set_case(self):
        finetune = _load_finetune_module()
        with tempfile.TemporaryDirectory() as test_dir:
            dataset = self._write_dataset(test_dir, 'diy', [1.0] * 4)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=self._collate)
            model = SimpleNamespace(global_rank=0, world_size=1, _validation_dl=loader,
                                    _validation_names=['diy_'])

            for off in (OmegaConf.create({'batch_size': 2}),
                        OmegaConf.create({'batch_size': 2, 'cost_batching': {'enabled': False}})):
                finetune.apply_cost_batch_sampler_to_validation(model, off)
                assert model._validation_dl is loader, "off must leave the dataloader untouched"

            # One validation set: NeMo hands over a bare DataLoader rather than a
            # list, and it has to come back as one or Lightning sees no val set.
            finetune.apply_cost_batch_sampler_to_validation(model, OmegaConf.create(
                {'batch_size': 2, 'shuffle': False, 'cost_batching': {'enabled': True, 'budget': 40.0}}
            ))
            assert isinstance(model._validation_dl, torch.utils.data.DataLoader)
            assert isinstance(model._validation_dl.batch_sampler, DurationCostBatchSampler)


class TestLossWeighting:
    """``model.loss_weighting``: a token's weight must not depend on its batch."""

    @staticmethod
    def _reduce(reduction, losses, normalizer=1.0):
        from nemo.collections.asr.losses.rnnt import RNNTLoss

        owner = SimpleNamespace(reduction=reduction, fixed_normalizer=normalizer)
        return RNNTLoss.reduce(owner, losses, torch.ones_like(losses, dtype=torch.long))

    @staticmethod
    def _weight_of_first_clip(reduction, batch_size, normalizer=1.0):
        losses = torch.ones(batch_size, requires_grad=True)
        TestLossWeighting._reduce(reduction, losses, normalizer).backward()
        return losses.grad[0].item()

    @pytest.mark.unit
    def test_sum_fixed_divides_the_sum_by_the_constant(self):
        reduced = self._reduce("sum_fixed", torch.tensor([1.0, 2.0, 5.0]), normalizer=4.0)
        assert reduced.item() == pytest.approx(2.0)

    @pytest.mark.unit
    def test_sum_fixed_weighs_a_clip_the_same_alone_or_among_sixteen(self):
        # The defect being fixed: mean_batch gives a lone clip 1 and one of 16 only 1/16.
        assert self._weight_of_first_clip("mean_batch", 1) == pytest.approx(1.0)
        assert self._weight_of_first_clip("mean_batch", 16) == pytest.approx(1 / 16)
        alone = self._weight_of_first_clip("sum_fixed", 1, normalizer=6.2)
        crowded = self._weight_of_first_clip("sum_fixed", 16, normalizer=6.2)
        assert alone == pytest.approx(crowded) == pytest.approx(1 / 6.2)

    @staticmethod
    def _fake_model(sampler_mean=None, batch_size=4):
        loss = SimpleNamespace(reduction="mean_batch", fixed_normalizer=1.0)
        if sampler_mean is None:
            dataloader = SimpleNamespace(batch_sampler=None, batch_size=batch_size)
        else:
            stats = {"clips_per_batch_mean": sampler_mean}
            dataloader = SimpleNamespace(batch_sampler=SimpleNamespace(stats=lambda: stats), batch_size=None)
        seen = []

        def training_step(batch, batch_nb):
            seen.append(loss.reduction)
            return {"loss": batch}

        return SimpleNamespace(loss=loss, _train_dl=dataloader, training_step=training_step), seen

    @staticmethod
    def _cfg(mode):
        block = {} if mode is None else {"loss_weighting": {"mode": mode}}
        return OmegaConf.create({"model": {**block, "train_ds": {"batch_size": 4}}})

    @pytest.mark.unit
    def test_per_token_switches_only_during_the_training_step(self):
        module = _load_finetune_module()
        model, seen = self._fake_model(sampler_mean=6.2)
        module.apply_loss_weighting(model, self._cfg("per_token"))
        assert model.loss.fixed_normalizer == pytest.approx(6.2)
        assert model.training_step("b", 0) == {"loss": "b"}
        assert seen == ["sum_fixed"]
        # Validation reuses the same loss object, so it must get mean_batch back.
        assert model.loss.reduction == "mean_batch"

    @pytest.mark.unit
    def test_per_token_without_cost_batching_uses_the_batch_size(self):
        module = _load_finetune_module()
        model, _ = self._fake_model(sampler_mean=None, batch_size=8)
        module.apply_loss_weighting(model, self._cfg("per_token"))
        assert model.loss.fixed_normalizer == pytest.approx(8.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("mode", [None, "per_batch"])
    def test_per_batch_and_absent_block_leave_nemo_untouched(self, mode):
        module = _load_finetune_module()
        model, seen = self._fake_model(sampler_mean=6.2)
        original = model.training_step
        module.apply_loss_weighting(model, self._cfg(mode))
        assert model.training_step is original
        assert model.loss.fixed_normalizer == 1.0

    @pytest.mark.unit
    def test_unknown_mode_is_refused(self):
        module = _load_finetune_module()
        model, _ = self._fake_model(sampler_mean=6.2)
        with pytest.raises(ValueError, match="per_token or per_batch"):
            module.apply_loss_weighting(model, self._cfg("per_clip"))
