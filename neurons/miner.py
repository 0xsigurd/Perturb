from __future__ import annotations

import argparse
import time
import typing

import bittensor as bt
import torch
import torch.nn.functional as F

from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import load_efficientnet_b5, resolve_target_index
from perturbnet.protocol import AttackChallenge


class PerturbMiner:
    def __init__(self, config: bt.config) -> None:
        self.config = config
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_efficientnet_b5(self.device)

        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

    def sync(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)

    async def forward(self, synapse: AttackChallenge) -> AttackChallenge:
        if synapse.norm_type != "Linf":
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        clean = decode_image_b64(synapse.clean_image_b64).to(self.device)
        target_index = resolve_target_index(synapse.true_label)
        if target_index is None:
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        epsilon = float(synapse.epsilon)
        min_delta = float(getattr(synapse, "min_delta", 0.002))

        # Basic default algorithm: small-step untargeted PGD.
        steps = 10
        step_size = max(epsilon / 4.0, 1.0 / 255.0)
        adv = clean.clone().detach()
        best = adv.clone()
        best_delta = 0.0
        for _ in range(steps):
            adv.requires_grad_(True)
            logits = self.model(adv.unsqueeze(0))
            loss = F.cross_entropy(logits, torch.tensor([target_index], device=self.device))
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + step_size * grad.sign()
            adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

            with torch.no_grad():
                pred = int(self.model(adv.unsqueeze(0)).argmax(dim=1).item())
            delta = float((adv - clean).abs().max().item())
            if delta > best_delta:
                best = adv.clone()
                best_delta = delta
            if pred != target_index and delta >= min_delta:
                best = adv.clone()
                break

        adv = best
        synapse.perturbed_image_b64 = encode_image_b64(adv)
        return synapse

    async def blacklist(self, synapse: AttackChallenge) -> typing.Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return True, "Missing caller hotkey"

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            return True, "Unregistered caller"

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            return True, "Caller is not validator"

        return False, "OK"

    async def priority(self, synapse: AttackChallenge) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return 0.0
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[uid])

    def run(self) -> None:
        self.sync()

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Miner hotkey is not registered on this netuid.")

        bt.logging.info("Serving miner axon...")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        bt.logging.info("Miner started. Waiting for validator queries.")
        while True:
            time.sleep(12)
            self.sync()


def build_config() -> bt.config:
    parser = argparse.ArgumentParser(description="Perturb subnet miner (default baseline)")
    parser.add_argument("--netuid", type=int, required=True)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    return bt.config(parser)


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()

