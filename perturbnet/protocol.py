from __future__ import annotations

import typing

import bittensor as bt

from perturbnet.constants import MODEL_NAME


class AttackChallenge(bt.Synapse):
    """
    Validator -> Miner challenge payload.

    Miner output is only `perturbed_image_b64`.
    """

    task_id: str
    model_name: str = MODEL_NAME
    prompt: str
    clean_image_b64: str
    true_label: str
    epsilon: float = 0.12
    norm_type: str = "Linf"
    min_delta: float = 0.002
    timeout_seconds: int = 60

    # Miner response payload.
    perturbed_image_b64: typing.Optional[str] = None

    def deserialize(self) -> typing.Optional[str]:
        return self.perturbed_image_b64

