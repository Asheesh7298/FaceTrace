"""
blockchain/anchor.py
Anchors pipeline results on-chain using web3.py.

Supports:
  - Hardhat local node (http://127.0.0.1:8545)
  - Sepolia testnet (via Infura)

Payload anchored:
  SHA-256 hash of a JSON blob containing:
    face_hash, content_hash, source_url,
    ensemble_score, engines_agreed, platform,
    exif_timestamp, wayback_first_seen, pipeline_version
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PIPELINE_VERSION = "1.0.0"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def sha256_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()


def sha256_of_str(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


def build_payload(
    face_hash: str,
    content_hash: str,
    source_url: str,
    ensemble_score: float,
    engines_agreed: int,
    platform: str,
    exif_timestamp: Optional[str],
    wayback_first_seen: Optional[str],
    match_found: bool,
    person_name: Optional[str] = None,
    identity_verified: bool = False,
    verification_score: float = 0.0,
    matched_image_url: Optional[str] = None,
) -> dict:
    """Builds the JSON payload that gets hashed and anchored.

    The SHA-256 of this exact JSON is what goes on-chain, so it captures not
    just *that* content existed but *who* the pipeline identified, how it was
    verified, and how confident it was — a self-describing forensic record.
    """
    return {
        "face_hash":          face_hash,
        "content_hash":       content_hash,
        "source_url":         source_url,
        "matched_image_url":  matched_image_url or "none",
        "person_name":        person_name or "unidentified",
        "identity_verified":  identity_verified,
        "verification_score": round(verification_score, 4),
        "ensemble_score":     round(ensemble_score, 4),
        "engines_agreed":     engines_agreed,
        "platform":           platform,
        "exif_timestamp":     exif_timestamp or "unknown",
        "wayback_first_seen": wayback_first_seen or "unknown",
        "match_found":        match_found,
        "pipeline_version":   PIPELINE_VERSION,
        "anchored_at_utc":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ─── Blockchain Client ────────────────────────────────────────────────────────

class BlockchainAnchor:
    def __init__(self, network: str = "localhost"):
        """
        network: "localhost" (Hardhat) or "sepolia"
        """
        self.network = network
        self.w3 = self._connect()
        self.contract = self._load_contract()
        self.account = self._load_account()

    def _connect(self) -> Web3:
        if self.network == "localhost":
            rpc = "http://127.0.0.1:8545"
            logger.info("Connecting to Hardhat local node...")
        elif self.network == "sepolia":
            rpc = os.getenv("INFURA_SEPOLIA_URL", "")
            if not rpc:
                raise ValueError("INFURA_SEPOLIA_URL not set in .env")
            logger.info("Connecting to Sepolia via Infura...")
        else:
            raise ValueError(f"Unknown network: {self.network}")

        w3 = Web3(Web3.HTTPProvider(rpc))
        if not w3.is_connected():
            raise ConnectionError(f"Cannot connect to {rpc}")

        logger.info(f"Connected ✓ — chain ID: {w3.eth.chain_id}, "
                    f"block: {w3.eth.block_number}")
        return w3

    def _load_contract(self):
        deployment_file = Path(__file__).parent.parent / f"deployment_{self.network}.json"
        if not deployment_file.exists():
            raise FileNotFoundError(
                f"No deployment found at {deployment_file}.\n"
                f"Run: npm run deploy:{'local' if self.network == 'localhost' else self.network}"
            )

        deployment = json.loads(deployment_file.read_text())
        self.contract_address = deployment["contractAddress"]
        abi = deployment["abi"]

        contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contract_address),
            abi=abi,
        )
        logger.info(f"Contract loaded: {self.contract_address}")
        return contract

    def _load_account(self):
        if self.network == "localhost":
            # Hardhat pre-funded accounts — use account 0
            accounts = self.w3.eth.accounts
            if not accounts:
                raise ValueError("No accounts on Hardhat node — is it running?")
            account_addr = accounts[0]
            logger.info(f"Using Hardhat account: {account_addr}")

            balance = self.w3.eth.get_balance(account_addr)
            logger.info(f"Balance: {Web3.from_wei(balance, 'ether'):.2f} ETH")
            return account_addr  # For local, we use send_transaction (no signing needed)

        elif self.network == "sepolia":
            private_key = os.getenv("DEPLOYER_PRIVATE_KEY", "")
            if not private_key:
                raise ValueError("DEPLOYER_PRIVATE_KEY not set in .env")
            if not private_key.startswith("0x"):
                private_key = "0x" + private_key

            acct = Account.from_key(private_key)
            logger.info(f"Using Sepolia account: {acct.address}")

            balance = self.w3.eth.get_balance(acct.address)
            logger.info(f"Balance: {Web3.from_wei(balance, 'ether'):.6f} ETH")

            if balance == 0:
                raise ValueError(
                    "Account has 0 ETH on Sepolia — get faucet ETH from sepoliafaucet.com"
                )
            return acct

    def anchor(self, payload: dict) -> dict:
        """
        Anchors the payload on-chain.
        Returns tx receipt + payload hash.
        """
        payload_json   = json.dumps(payload, sort_keys=True)
        payload_hash   = sha256_of_str(payload_json)
        confidence_int = int(payload["ensemble_score"] * 100)

        logger.info(f"\nAnchoring payload hash: {payload_hash}")
        logger.info(f"  Face hash:    {payload['face_hash']}")
        logger.info(f"  Content hash: {payload['content_hash']}")
        logger.info(f"  Source URL:   {payload['source_url']}")
        logger.info(f"  Confidence:   {payload['ensemble_score']}")
        logger.info(f"  Match found:  {payload['match_found']}")

        tx_hash = None
        receipt = None

        if self.network == "localhost":
            # Hardhat: use send_transaction (no signing needed)
            tx = self.contract.functions.anchor(
                payload_hash,
                payload["face_hash"],
                payload["content_hash"],
                payload["source_url"],
                confidence_int,
                payload["match_found"],
            ).transact({"from": self.account})

            receipt = self.w3.eth.wait_for_transaction_receipt(tx)
            tx_hash = receipt.transactionHash.hex()

        elif self.network == "sepolia":
            # Sepolia: sign and send
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            gas_price = self.w3.eth.gas_price

            tx = self.contract.functions.anchor(
                payload_hash,
                payload["face_hash"],
                payload["content_hash"],
                payload["source_url"],
                confidence_int,
                payload["match_found"],
            ).build_transaction({
                "from":     self.account.address,
                "nonce":    nonce,
                "gasPrice": gas_price,
            })

            signed = self.account.sign_transaction(tx)
            sent   = self.w3.eth.send_raw_transaction(signed.rawTransaction)

            logger.info(f"Transaction sent: {sent.hex()}")
            logger.info("Waiting for confirmation...")
            receipt = self.w3.eth.wait_for_transaction_receipt(sent, timeout=120)
            tx_hash = receipt.transactionHash.hex()

        logger.info(f"\n✅ Anchored on-chain!")
        logger.info(f"   TX hash:  {tx_hash}")
        logger.info(f"   Block:    {receipt.blockNumber}")
        logger.info(f"   Gas used: {receipt.gasUsed}")

        if self.network == "sepolia":
            logger.info(f"   Etherscan: https://sepolia.etherscan.io/tx/{tx_hash}")

        return {
            "payload_hash":      payload_hash,
            "tx_hash":           tx_hash,
            "block_number":      receipt.blockNumber,
            "gas_used":          receipt.gasUsed,
            "contract_address":  self.contract_address,
            "network":           self.network,
            "etherscan_url": (
                f"https://sepolia.etherscan.io/tx/{tx_hash}"
                if self.network == "sepolia" else None
            ),
        }

    def verify(self, payload_hash: str) -> dict:
        """
        Fetches on-chain record and verifies it matches the given hash.
        """
        logger.info(f"\nVerifying payload hash: {payload_hash}")

        record = self.contract.functions.getProof(payload_hash).call()

        # record is a tuple: (payloadHash, faceHash, contentHash, sourceUrl,
        #                      confidenceScore, timestamp, matchFound)
        on_chain_hash = record[0]

        if not on_chain_hash:
            logger.error("❌ Hash NOT found on-chain")
            return {"verified": False, "error": "Hash not found on-chain"}

        is_match = on_chain_hash == payload_hash

        result = {
            "verified":          is_match,
            "payload_hash":      payload_hash,
            "on_chain_hash":     on_chain_hash,
            "face_hash":         record[1],
            "content_hash":      record[2],
            "source_url":        record[3],
            "confidence":        int(record[4]) / 100,
            "anchored_at":       time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(int(record[5]))
            ),
            "match_found":       record[6],
        }

        if is_match:
            logger.info("✅ VERIFIED — on-chain hash matches local hash")
        else:
            logger.error("❌ MISMATCH — on-chain hash differs from local hash")

        return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python blockchain/anchor.py <result_json_path> [network]")
        print("  network: localhost (default) or sepolia")
        sys.exit(1)

    result_path = sys.argv[1]
    network     = sys.argv[2] if len(sys.argv) > 2 else "localhost"

    pipeline_result = json.loads(Path(result_path).read_text())

    best = pipeline_result.get("search_results", {}).get("best_match") or {}

    payload = build_payload(
        face_hash          = pipeline_result["face"]["face_hash"],
        content_hash       = best.get("content_hash", "none"),
        source_url         = best.get("url", "none"),
        ensemble_score     = best.get("composite_score", 0.0),
        engines_agreed     = len(best.get("engines", [])),
        platform           = best.get("platform", "unknown"),
        exif_timestamp     = pipeline_result["face"].get("exif", {}).get("timestamp"),
        wayback_first_seen = best.get("wayback", {}).get("first_seen"),
        match_found        = pipeline_result["search_results"]["success"],
    )

    anchor_client = BlockchainAnchor(network=network)

    print("\n--- ANCHORING ---")
    anchor_result = anchor_client.anchor(payload)
    print(json.dumps(anchor_result, indent=2))

    print("\n--- VERIFYING ---")
    verify_result = anchor_client.verify(anchor_result["payload_hash"])
    print(json.dumps(verify_result, indent=2))
