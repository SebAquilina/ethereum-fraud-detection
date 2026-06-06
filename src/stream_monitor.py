"""
Live block monitor. Connects to an Ethereum node (WebSocket or HTTP), watches
for new blocks and scores the addresses it sees as they come in.

How it hangs together:
  - a listener polls for new blocks roughly once a second
  - on_transaction fires straight away per tx, so the UI can show it before
    the score is ready
  - a small pool of scorer threads does the actual Etherscan + model work
  - on_result fires once a score lands, and the UI fills the row in

Needs web3 (pip install web3).
"""
import os
import sys
import time
import json
import threading
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Callable, Optional, Dict, List, Set
from queue import Queue, Empty

from web3 import Web3

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from live_detector import FraudDetector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stream_monitor")


# ---------------------------------------------------------------------------
# LRU Cache for scored addresses
# ---------------------------------------------------------------------------
class AddressCache:
    """Simple LRU cache to avoid re-scoring recently seen addresses."""

    def __init__(self, max_size: int = 10_000):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, address: str) -> Optional[dict]:
        address = address.lower()
        with self._lock:
            if address in self._cache:
                self._cache.move_to_end(address)
                return self._cache[address]
        return None

    def put(self, address: str, result: dict):
        address = address.lower()
        with self._lock:
            self._cache[address] = result
            self._cache.move_to_end(address)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def __contains__(self, address: str):
        with self._lock:
            return address.lower() in self._cache

    def __len__(self):
        with self._lock:
            return len(self._cache)


# ---------------------------------------------------------------------------
# Stream Monitor
# ---------------------------------------------------------------------------
class StreamMonitor:
    """Watches the chain, pulls addresses out of each new block and scores them.

    The two callbacks fire at different times: on_transaction the moment a block
    arrives, on_result once that address has actually been scored.
    """

    ALERT_LEVELS = {"CRITICAL", "HIGH"}
    NUM_SCORERS = 3  # how many addresses we score in parallel

    def __init__(
        self,
        provider_uri: Optional[str] = None,
        fraud_threshold: float = 0.7,
        max_addresses_per_block: int = 50,
        on_alert: Optional[Callable[[dict], None]] = None,
        on_result: Optional[Callable[[dict], None]] = None,
        on_transaction: Optional[Callable[[dict], None]] = None,
    ):
        self.provider_uri = (
            provider_uri
            or os.getenv("ETHEREUM_WS_URL")
            or os.getenv("ETHEREUM_HTTP_URL")
        )
        if not self.provider_uri:
            raise ValueError(
                "No provider URI. Set ETHEREUM_WS_URL or ETHEREUM_HTTP_URL, "
                "or pass --provider. Get a free key from https://www.alchemy.com/ "
                "or https://infura.io/"
            )

        self.fraud_threshold = fraud_threshold
        self.max_addresses_per_block = max_addresses_per_block
        self.on_alert = on_alert
        self.on_result = on_result
        self.on_transaction = on_transaction

        # Connect to Ethereum
        if self.provider_uri.startswith("ws"):
            self.w3 = Web3(Web3.WebsocketProvider(self.provider_uri))
        else:
            self.w3 = Web3(Web3.HTTPProvider(self.provider_uri))

        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to Ethereum node at {self.provider_uri}")
        log.info("Connected to Ethereum  (chain_id=%s)", self.w3.eth.chain_id)

        # Load fraud detection models
        self.detector = FraudDetector()
        if not self.detector.models_loaded:
            raise RuntimeError("Fraud detection models failed to load")

        # State
        self.cache = AddressCache(max_size=10_000)
        self._scoring_queue: Queue = Queue()
        self._running = False
        self._stats = {
            "blocks_processed": 0,
            "transactions_seen": 0,
            "addresses_scored": 0,
            "alerts_raised": 0,
            "start_time": None,
        }
        self._alerts: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, blocking: bool = True):
        self._running = True
        self._stats["start_time"] = datetime.now(timezone.utc).isoformat()
        log.info("Starting stream monitor (threshold=%.0f%%, scorers=%d)",
                 self.fraud_threshold * 100, self.NUM_SCORERS)

        # Start multiple scorer threads
        for i in range(self.NUM_SCORERS):
            t = threading.Thread(target=self._scorer_loop, daemon=True,
                                 name=f"scorer-{i}")
            t.start()

        if blocking:
            self._block_listener()
        else:
            threading.Thread(
                target=self._block_listener, daemon=True, name="block-listener"
            ).start()

    def stop(self):
        log.info("Stopping stream monitor...")
        self._running = False

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def recent_alerts(self) -> List[dict]:
        return list(self._alerts[-100:])

    # ------------------------------------------------------------------
    # Internal: block listener
    # ------------------------------------------------------------------

    def _block_listener(self):
        last_block = None

        while self._running:
            try:
                current_block = self.w3.eth.block_number

                if last_block is None:
                    # Process the current block immediately so the UI
                    # is populated the moment the page loads.
                    last_block = current_block - 1
                    log.info("Processing current block %d immediately", current_block)

                while last_block < current_block and self._running:
                    last_block += 1
                    self._process_block(last_block)

                # Poll every 1s for minimal latency
                time.sleep(1)

            except Exception as e:
                log.error("Block listener error: %s", e)
                time.sleep(3)

    def _process_block(self, block_number: int):
        try:
            block = self.w3.eth.get_block(block_number, full_transactions=True)
        except Exception as e:
            log.warning("Failed to fetch block %d: %s", block_number, e)
            return

        txs = block.get("transactions", [])
        self._stats["transactions_seen"] += len(txs)
        self._stats["blocks_processed"] += 1

        block_ts = datetime.fromtimestamp(
            block.get("timestamp", 0), tz=timezone.utc
        ).isoformat()

        # Collect unique addresses for scoring
        addresses: Set[str] = set()
        tx_map: Dict[str, dict] = {}

        for tx in txs:
            from_addr = (tx.get("from") or "").lower()
            to_addr = (tx.get("to") or "").lower()
            value_wei = tx.get("value", 0)
            value_eth = value_wei / 1e18 if value_wei else 0

            tx_hash = tx.get("hash", b"")
            if isinstance(tx_hash, bytes):
                tx_hash = tx_hash.hex()
            else:
                tx_hash = str(tx_hash)

            tx_info = {
                "hash": tx_hash,
                "block": block_number,
                "value_eth": round(value_eth, 6),
                "timestamp": block_ts,
            }

            if from_addr and from_addr not in self.cache:
                addresses.add(from_addr)
                tx_map[from_addr] = tx_info
            if to_addr and to_addr not in self.cache:
                addresses.add(to_addr)
                tx_map[to_addr] = tx_info

        # Prioritize higher-value, cap per block
        sorted_addrs = sorted(
            addresses,
            key=lambda a: tx_map.get(a, {}).get("value_eth", 0),
            reverse=True,
        )

        selected = sorted_addrs[: self.max_addresses_per_block]

        # Broadcast only the addresses we will actually score
        if self.on_transaction:
            for addr in selected:
                try:
                    self.on_transaction({
                        "type": "transaction",
                        "address": addr,
                        "direction": "sent" if tx_map[addr].get("value_eth", 0) > 0 else "received",
                        "tx": tx_map[addr],
                    })
                except Exception:
                    pass

        for addr in selected:
            self._scoring_queue.put((addr, tx_map.get(addr, {})))

        if txs:
            log.info(
                "Block %d: %d txs, %d queued for scoring (queue: %d)",
                block_number, len(txs),
                min(len(sorted_addrs), self.max_addresses_per_block),
                self._scoring_queue.qsize(),
            )

    # ------------------------------------------------------------------
    # Internal: scorer loop (runs in multiple threads)
    # ------------------------------------------------------------------

    def _scorer_loop(self):
        while self._running:
            try:
                addr, tx_info = self._scoring_queue.get(timeout=2)
            except Empty:
                continue

            if self.cache.get(addr):
                continue

            try:
                result = self.detector.score_address(addr)
            except Exception as e:
                log.error("Scoring failed for %s: %s", addr, e)
                continue

            if result.get("error"):
                self.cache.put(addr, result)
                continue

            result["stream_context"] = {
                "triggering_tx": tx_info,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }

            self.cache.put(addr, result)
            self._stats["addresses_scored"] += 1

            # Broadcast scored result
            if self.on_result:
                try:
                    self.on_result(result)
                except Exception:
                    pass

            prob = result.get("fraud_probability", 0) or 0
            risk = result.get("risk_level", "MINIMAL")

            if prob >= self.fraud_threshold or risk in self.ALERT_LEVELS:
                self._stats["alerts_raised"] += 1
                self._alerts.append(result)
                log.warning(
                    "ALERT  %s — fraud_prob=%.2f%% risk=%s",
                    addr, prob * 100, risk,
                )
                if self.on_alert:
                    try:
                        self.on_alert(result)
                    except Exception:
                        pass
            else:
                log.info("OK     %s — fraud_prob=%.2f%%", addr, prob * 100)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Ethereum fraud stream monitor")
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max-per-block", type=int, default=20)
    args = parser.parse_args()

    provider = (args.provider
                or os.getenv("ETHEREUM_WS_URL")
                or os.getenv("ETHEREUM_HTTP_URL"))

    def print_alert(result):
        addr = result["address"]
        prob = result["fraud_probability"]
        risk = result["risk_level"]
        tx = result.get("stream_context", {}).get("triggering_tx", {})
        print(f"\n{'!'*60}")
        print(f"  FRAUD ALERT: {addr}")
        print(f"  Probability: {prob:.2%}  |  Risk: {risk}")
        print(f"  TX: {tx.get('hash', 'N/A')}")
        print(f"{'!'*60}\n")

    def print_tx(data):
        addr = data["address"]
        tx = data["tx"]
        print(f"  TX  {addr[:20]}...  {tx['value_eth']} ETH  ({data['direction']})")

    monitor = StreamMonitor(
        provider_uri=provider,
        fraud_threshold=args.threshold,
        max_addresses_per_block=args.max_per_block,
        on_alert=print_alert,
        on_transaction=print_tx,
    )

    try:
        monitor.start(blocking=True)
    except KeyboardInterrupt:
        monitor.stop()
        print("\nMonitor stopped. Stats:", json.dumps(monitor.stats, indent=2))
