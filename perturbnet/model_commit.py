"""On-chain model commitment schema."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

MODEL_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
ARCHITECTURE = "efficientnet_v2_l"
NUM_CLASSES = 1000

CHAIN_COMMIT_MAX_BYTES = 128
CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS = 44
MODEL_HASH_CHARS = 16


class MinerChainCommit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hf_repo_id: str | None = Field(default=None, alias="r")
    hf_revision: str | None = Field(default=None, alias="rv")
    model_hash: str | None = Field(default=None, alias="h")

    @property
    def repo_at_revision(self) -> str:
        return f"{self.hf_repo_id}@{self.hf_revision}"


def serialize_chain_commit(status: MinerChainCommit) -> tuple[dict, str]:
    data_dict = status.model_dump(by_alias=True, exclude_none=True)
    return data_dict, json.dumps(data_dict, separators=(",", ":"))


def validate_chain_commit_payload(
    status: MinerChainCommit,
    max_bytes: int = CHAIN_COMMIT_MAX_BYTES,
    max_hf_repo_id_chars: int = CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS,
) -> tuple[dict, str]:
    data_dict, data = serialize_chain_commit(status)
    if status.hf_repo_id and len(status.hf_repo_id) > max_hf_repo_id_chars:
        raise ValueError(
            f"HF repo id is too long for the chain payload budget: {len(status.hf_repo_id)} > {max_hf_repo_id_chars}"
        )
    payload_bytes = len(data.encode())
    if payload_bytes > max_bytes:
        raise ValueError(f"Chain commit exceeds payload budget: {payload_bytes} > {max_bytes} bytes")
    return data_dict, data


def parse_chain_commit(data: str) -> MinerChainCommit | None:
    try:
        payload = json.loads(data)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        commit = MinerChainCommit.model_validate(payload)
    except Exception:
        return None
    if not commit.hf_repo_id or not commit.hf_revision or not commit.model_hash:
        return None
    return commit


@dataclass(frozen=True)
class RepoRevision:
    repo_id: str
    revision: str

    def __str__(self) -> str:
        return f"{self.repo_id}@{self.revision}"


def parse_repo_at_revision(text: str) -> RepoRevision | None:
    repo_id, sep, revision = str(text or "").strip().partition("@")
    if not sep or not repo_id or not revision or "/" not in repo_id:
        return None
    return RepoRevision(repo_id=repo_id, revision=revision)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_model_and_hotkey(path: str | os.PathLike[str], hotkey: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    digest.update(hotkey.encode("utf-8"))
    return digest.hexdigest()
