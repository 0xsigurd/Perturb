from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Sequence

import bittensor as bt
import numpy as np
import requests
import torch

from perturbnet import constants as C
from perturbnet.image_io import decode_image_b64
from perturbnet.model import load_efficientnet_b5, normalize_prediction_label, predict_label
from perturbnet.protocol import AttackChallenge


@dataclass
class ChallengeSpec:
    task_id: str
    model_name: str
    prompt: str
    clean_image_b64: str
    true_label: str
    epsilon: float
    norm_type: str
    timeout_seconds: int


class PerturbValidator:
    def __init__(self, config: bt.config) -> None:
        self.config = config
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_efficientnet_b5(self.device)
        self.step = 0
        self.last_weight_block = 0
        self.state_path = os.path.join(self.config.logging.logging_dir, C.VALIDATOR_STATE_FILENAME)

        self.processed_counts = np.zeros(int(self.metagraph.n), dtype=np.int32)
        self.score_histories: list[list[float]] = [[] for _ in range(int(self.metagraph.n))]

        self._load_state()

    def sync(self) -> None:
        old_n = int(self.metagraph.n)
        self.metagraph.sync(subtensor=self.subtensor)
        new_n = int(self.metagraph.n)
        if new_n != old_n:
            resized_counts = np.zeros(new_n, dtype=np.int32)
            copied = min(len(self.processed_counts), new_n)
            resized_counts[:copied] = self.processed_counts[:copied]
            self.processed_counts = resized_counts
            if new_n > len(self.score_histories):
                self.score_histories.extend([[] for _ in range(new_n - len(self.score_histories))])
            else:
                self.score_histories = self.score_histories[:new_n]

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        with open(self.state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        self.step = int(state.get("step", 0))
        self.last_weight_block = int(state.get("last_weight_block", 0))

        saved_counts = state.get("processed_counts", [])
        copied = min(len(saved_counts), len(self.processed_counts))
        for idx in range(copied):
            self.processed_counts[idx] = int(saved_counts[idx])

        saved_histories = state.get("score_histories", [])
        copied_h = min(len(saved_histories), len(self.score_histories))
        for idx in range(copied_h):
            raw = saved_histories[idx]
            if isinstance(raw, list):
                self.score_histories[idx] = [float(x) for x in raw[-self.config.perturb.history_size :]]

    def _save_state(self) -> None:
        directory = os.path.dirname(self.state_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "step": int(self.step),
            "last_weight_block": int(self.last_weight_block),
            "processed_counts": self.processed_counts.tolist(),
            "score_histories": [history[-self.config.perturb.history_size :] for history in self.score_histories],
        }
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def _seed_from_block(self, block: int) -> int:
        digest = hashlib.sha256(f"{C.SUBNET_NAMESPACE}:{self.config.netuid}:{block}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _sample_epsilon(self, seed: int) -> float:
        # Deterministic epsilon in [0.06, 0.2]
        return 0.06 + (seed % 1400) / 10000.0

    def _choose_prompt(self, seed: int) -> str:
        rng = random.Random(seed)
        return rng.choice(list(C.PROMPTS))

    def _parse_llm_endpoint_result(self, payload: Any) -> bool | None:
        if isinstance(payload, bool):
            return payload
        if not isinstance(payload, dict):
            return None

        for key in ("is_match", "match", "ok", "valid"):
            value = payload.get(key)
            if isinstance(value, bool):
                return value
        return None

    def _llm_endpoint_check(self, predicted_label: str, expected_label: str) -> bool:
        endpoint = str(
            getattr(
                self.config.perturb,
                "llm_endpoint_url",
                getattr(self.config.perturb, "label_match_endpoint", ""),
            )
            or ""
        ).strip()
        normalized_prediction = normalize_prediction_label(predicted_label)
        if not endpoint:
            bt.logging.error("LLM endpoint url is empty; rejecting verification check.")
            return False

        payload = {
            "prediction": normalized_prediction,
            "target_label": expected_label,
            "llm_model": str(
                getattr(
                    self.config.perturb,
                    "llm_endpoint_model",
                    getattr(self.config.perturb, "label_match_model", C.LLM_ENDPOINT_MODEL),
                )
            ),
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=8)
            response.raise_for_status()
            parsed = self._parse_llm_endpoint_result(response.json())
            if parsed is None:
                bt.logging.error("LLM endpoint returned unrecognized payload shape; rejecting check.")
                return False
            return bool(parsed)
        except Exception as exc:
            bt.logging.error("LLM endpoint request failed (%s); rejecting check.", exc)
            return False

    def _fetch_image_for_prompt(self, prompt: str, seed: int) -> str:
        endpoint = self.config.perturb.image_endpoint
        params = {
            "prompt": prompt,
            "seed": seed,
            "image_size": self.config.perturb.image_size,
            "random_mode": "true",
        }
        response = requests.get(endpoint, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
        image_b64 = str(data.get("image_base64", ""))
        if not image_b64:
            raise ValueError("image response missing image_base64")
        return image_b64

    def _load_fallback_image_b64(self) -> str:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fallback_path = os.path.join(project_root, C.FALLBACK_IMAGE_RELATIVE_PATH)
        with open(fallback_path, "rb") as handle:
            raw = handle.read()
        if not raw:
            raise ValueError(f"fallback image is empty: {fallback_path}")
        return base64.b64encode(raw).decode("utf-8")

    def generate_challenge(self, block: int) -> ChallengeSpec:
        model_name = C.MODEL_NAME
        base_seed = self._seed_from_block(block)
        for attempt in range(self.config.perturb.max_challenge_attempts):
            seed = base_seed + attempt
            chosen_prompt = self._choose_prompt(seed)
            try:
                image_b64 = self._fetch_image_for_prompt(prompt=chosen_prompt, seed=seed)
                effective_prompt = chosen_prompt
            except Exception as exc:
                bt.logging.warning("Challenge image fetch failed (%s), using fallback dog image.", exc)
                try:
                    image_b64 = self._load_fallback_image_b64()
                    effective_prompt = C.FALLBACK_LABEL
                except Exception as fallback_exc:
                    bt.logging.warning("Fallback image load failed, retrying: %s", fallback_exc)
                    continue

            epsilon = self._sample_epsilon(seed)
            task_id = f"{block}-{seed}"

            try:
                image = decode_image_b64(image_b64).to(self.device)
                predicted = predict_label(self.model, image)
                predicted_label = normalize_prediction_label(predicted)
            except Exception as exc:
                bt.logging.warning("Challenge decode/model validation failed, retrying: %s", exc)
                continue

            # Verify the candidate by semantically checking model output against the API prompt label.
            if not self._llm_endpoint_check(predicted_label, effective_prompt):
                bt.logging.debug(
                    "Challenge rejected by classifier consistency: pred=%s expected=%s",
                    predicted_label,
                    effective_prompt,
                )
                continue

            return ChallengeSpec(
                task_id=task_id,
                model_name=model_name,
                prompt=effective_prompt,
                clean_image_b64=image_b64,
                # Use exact EfficientNet class label for miner targeting and response verification.
                true_label=predicted_label,
                epsilon=epsilon,
                norm_type="Linf",
                timeout_seconds=self.config.perturb.timeout_seconds,
            )

        raise RuntimeError("Unable to build a validated challenge after max attempts")

    def _available_miner_uids(self) -> list[int]:
        my_hotkey = self.wallet.hotkey.ss58_address
        uids: list[int] = []
        for uid in range(int(self.metagraph.n)):
            if self.metagraph.hotkeys[uid] == my_hotkey:
                continue
            if self.metagraph.axons[uid].ip == "0.0.0.0":
                continue
            uids.append(uid)
        return uids

    def _valuable_miner_uids(self, candidate_uids: Sequence[int]) -> list[int]:
        min_processed = int(self.config.perturb.min_processed_count)
        return [uid for uid in candidate_uids if int(self.processed_counts[uid]) > min_processed]

    def _select_random_miners(self, candidate_uids: Sequence[int], seed: int) -> list[int]:
        if not candidate_uids:
            return []
        valuable = self._valuable_miner_uids(candidate_uids)
        pool = list(valuable) if valuable else list(candidate_uids)
        k = min(int(self.config.perturb.k_miners), len(pool))
        rng = random.Random(seed)
        return sorted(rng.sample(pool, k=k))

    async def _query_miners(self, uids: Sequence[int], challenge: ChallengeSpec):
        axons = [self.metagraph.axons[uid] for uid in uids]
        synapse = AttackChallenge(
            task_id=challenge.task_id,
            model_name=challenge.model_name,
            prompt=challenge.prompt,
            clean_image_b64=challenge.clean_image_b64,
            true_label=challenge.true_label,
            epsilon=challenge.epsilon,
            norm_type=challenge.norm_type,
            min_delta=self.config.perturb.min_linf_delta,
            timeout_seconds=challenge.timeout_seconds,
        )
        responses = await self.dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=False,
            timeout=challenge.timeout_seconds,
        )
        return responses

    def verify_and_score(self, challenge: ChallengeSpec, perturbed_image_b64: str, response_time_ms: int) -> float:
        try:
            x_clean = decode_image_b64(challenge.clean_image_b64).to(self.device)
            x_adv = decode_image_b64(perturbed_image_b64).to(self.device)
        except Exception:
            return 0.0

        if x_adv.shape != x_clean.shape:
            return 0.0
        if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
            return 0.0

        if challenge.norm_type == "Linf":
            norm = (x_adv - x_clean).abs().max().item()
        elif challenge.norm_type == "L2":
            norm = float((x_adv - x_clean).norm(2).item())
        else:
            norm = float((x_adv - x_clean).ne(0).sum().item())

        if norm < self.config.perturb.min_linf_delta:
            return 0.0
        if norm > min(float(challenge.epsilon), float(self.config.perturb.max_linf_delta)):
            return 0.0

        prediction = predict_label(self.model, x_adv)
        # Successful perturbation means the response no longer verifies as original label.
        if self._llm_endpoint_check(prediction, challenge.true_label):
            return 0.0

        perturbation_ratio = norm / max(1e-12, float(challenge.epsilon))
        perturbation_score = 1.0 - min(perturbation_ratio, 1.0)

        time_ratio = response_time_ms / (challenge.timeout_seconds * 1000.0)
        speed_score = 1.0 - min(time_ratio, 1.0)

        return C.PERTURBATION_WEIGHT * perturbation_score + C.SPEED_WEIGHT * speed_score

    def _update_histories(self, uids: Sequence[int], rewards: Sequence[float]) -> None:
        history_size = int(self.config.perturb.history_size)
        for uid, reward in zip(uids, rewards):
            self.processed_counts[uid] += 1
            self.score_histories[uid].append(float(reward))
            if len(self.score_histories[uid]) > history_size:
                self.score_histories[uid] = self.score_histories[uid][-history_size:]

    def _rank_points(self, rank: int) -> float:
        # rank is 0-based among eligible miners sorted by avg score
        if rank == 0:
            return 50.0
        if rank == 1:
            return 30.0
        if rank == 2:
            return 10.0
        if rank <= 9:
            return 5.0
        return 3.0

    def _set_weights(self) -> None:
        eligible: list[tuple[int, float]] = []
        min_processed = int(self.config.perturb.min_processed_count)
        history_size = int(self.config.perturb.history_size)
        for uid in range(int(self.metagraph.n)):
            if self.processed_counts[uid] <= min_processed:
                continue
            history = self.score_histories[uid]
            if not history:
                continue
            tail = history[-history_size:]
            avg_score = float(sum(tail) / max(1, len(tail)))
            eligible.append((uid, avg_score))

        if not eligible:
            bt.logging.warning("No eligible miners with processed_count >= %d", min_processed)
            return

        eligible.sort(key=lambda x: (x[1], -x[0]), reverse=True)
        avg_raw = np.zeros(int(self.metagraph.n), dtype=np.float32)
        bonus_raw = np.zeros(int(self.metagraph.n), dtype=np.float32)
        for rank, (uid, avg_score) in enumerate(eligible):
            avg_raw[uid] = max(0.0, float(avg_score))
            bonus_raw[uid] = float(self._rank_points(rank))

        avg_total = float(avg_raw.sum())
        bonus_total = float(bonus_raw.sum())
        if avg_total <= 0.0 or bonus_total <= 0.0:
            bt.logging.warning("Average or rank-bonus totals are zero; skipping set_weights.")
            return

        avg_norm = avg_raw / avg_total
        bonus_norm = bonus_raw / bonus_total
        gamma = float(C.GAMMA_HISTORY_WEIGHT)
        raw = gamma * avg_norm + (1.0 - gamma) * bonus_norm

        total = float(raw.sum())
        if total <= 0.0:
            bt.logging.warning("Rank-based weights are all zero; skipping set_weights.")
            return

        normalized = raw / total
        uids = list(range(len(normalized)))
        weights = [float(v) for v in normalized.tolist()]
        ok, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )
        if ok:
            bt.logging.info("set_weights success")
        else:
            bt.logging.error("set_weights failed: %s", msg)

    def run(self) -> None:
        self.sync()
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Validator hotkey is not registered on this netuid.")

        tempo = self.subtensor.get_subnet_hyperparameters(self.config.netuid).tempo
        bt.logging.info("Validator started with tempo=%s", tempo)

        while True:
            try:
                self.sync()
                block = self.subtensor.get_current_block()
                challenge = self.generate_challenge(block=block)
                bt.logging.info(
                    "Challenge task=%s prompt=%s eps=%.4f",
                    challenge.task_id,
                    challenge.prompt,
                    challenge.epsilon,
                )

                available_uids = self._available_miner_uids()
                if not available_uids:
                    bt.logging.warning("No miners available")
                    time.sleep(self.config.perturb.query_interval_seconds)
                    continue
                valuable_uids = self._valuable_miner_uids(available_uids)
                miner_uids = self._select_random_miners(available_uids, seed=self._seed_from_block(block))
                if not miner_uids:
                    bt.logging.warning("Miner selection is empty")
                    time.sleep(self.config.perturb.query_interval_seconds)
                    continue
                bt.logging.info(
                    "Selected %d miners (valuable pool=%d, total pool=%d)",
                    len(miner_uids),
                    len(valuable_uids),
                    len(available_uids),
                )

                responses = asyncio.run(self._query_miners(miner_uids, challenge))
                rewards: list[float] = []
                for uid, response in zip(miner_uids, responses):
                    status_code = getattr(response.dendrite, "status_code", 0) if response.dendrite else 0
                    process_time = getattr(response.dendrite, "process_time", None) if response.dendrite else None
                    response_time_ms = int((process_time or challenge.timeout_seconds) * 1000)

                    if status_code != 200 or not response.perturbed_image_b64:
                        score = 0.0
                    else:
                        score = self.verify_and_score(
                            challenge=challenge,
                            perturbed_image_b64=response.perturbed_image_b64,
                            response_time_ms=response_time_ms,
                        )
                    rewards.append(score)
                    bt.logging.info(
                        "uid=%s status=%s score=%.6f processed=%d",
                        uid,
                        status_code,
                        score,
                        int(self.processed_counts[uid]) + 1,
                    )

                self._update_histories(miner_uids, rewards)
                self._save_state()

                blocks_since_weights = block - self.last_weight_block
                if blocks_since_weights >= tempo:
                    self._set_weights()
                    self.last_weight_block = block

                self.step += 1
                time.sleep(self.config.perturb.query_interval_seconds)
            except KeyboardInterrupt:
                bt.logging.info("Validator stopped by user.")
                break
            except Exception as exc:
                bt.logging.error("Validator loop error: %s", exc)
                time.sleep(5)


def build_config() -> bt.config:
    parser = argparse.ArgumentParser(description="Perturb subnet validator")
    parser.add_argument("--netuid", type=int, required=True)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)

    config = bt.config(parser)
    perturb_cfg = type("PerturbConfig", (), {})()
    config.perturb = perturb_cfg
    for key, value in C.VALIDATOR_CONFIG.items():
        setattr(config.perturb, key, value)
    return config


if __name__ == "__main__":
    validator = PerturbValidator(config=build_config())
    validator.run()

