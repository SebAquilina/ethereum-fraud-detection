"""
Turns the raw Etherscan payload for an address into the feature dict the
models expect. The column names and ordering have to line up exactly with
what was used at training time, hence the slightly awkward names below.
"""
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


def wei_to_ether(wei_str: str) -> float:
    """Wei string -> ether. Returns 0 on anything that won't parse."""
    try:
        return int(wei_str) / 1e18
    except (ValueError, TypeError):
        return 0.0


def compute_features(data: Dict[str, Any]) -> Dict[str, float]:
    """Build the full feature dict from one get_address_data() payload.
    Keys/order match the training columns."""
    address = data["address"].lower()
    normal_txs = data.get("normal_transactions", [])
    internal_txs = data.get("internal_transactions", [])
    erc20_txs = data.get("erc20_transfers", [])
    balance = data.get("balance", 0.0)
    
    # --- Normal transaction analysis ---
    sent_txs = [tx for tx in normal_txs if tx.get("from", "").lower() == address]
    received_txs = [tx for tx in normal_txs if tx.get("to", "").lower() == address]
    
    # Time-based features (using block timestamps)
    sent_times = sorted([int(tx.get("timeStamp", 0)) for tx in sent_txs])
    received_times = sorted([int(tx.get("timeStamp", 0)) for tx in received_txs])
    
    # Time differences in minutes
    def avg_time_between(times: List[int]) -> float:
        if len(times) < 2:
            return 0.0
        # Optimization: The sum of differences is equal to (last - first)
        return ((times[-1] - times[0]) / 60) / (len(times) - 1)
    
    # Values in Ether
    sent_values = [wei_to_ether(tx.get("value", "0")) for tx in sent_txs]
    received_values = [wei_to_ether(tx.get("value", "0")) for tx in received_txs]
    
    # Contract creation count
    contract_creates = sum(1 for tx in sent_txs if tx.get("to", "") == "" or tx.get("contractAddress", ""))
    
    # Unique addresses
    unique_from = len(set(tx.get("from", "").lower() for tx in received_txs if tx.get("from")))
    unique_to = len(set(tx.get("to", "").lower() for tx in sent_txs if tx.get("to")))
    
    # Contract-related values
    sent_to_contract_values = [
        wei_to_ether(tx.get("value", "0")) 
        for tx in sent_txs 
        if tx.get("contractAddress") or tx.get("input", "0x") != "0x"
    ]
    
    # Total Ether calculations
    total_sent = sum(sent_values) if sent_values else 0.0
    total_received = sum(received_values) if received_values else 0.0
    total_sent_contracts = sum(sent_to_contract_values) if sent_to_contract_values else 0.0
    
    # Time span
    all_times = sorted(sent_times + received_times)
    time_diff_mins = (all_times[-1] - all_times[0]) / 60 if len(all_times) >= 2 else 0.0
    
    # --- ERC20 analysis ---
    erc20_received = [tx for tx in erc20_txs if tx.get("to", "").lower() == address]
    erc20_sent = [tx for tx in erc20_txs if tx.get("from", "").lower() == address]
    erc20_to_contract = [tx for tx in erc20_sent if tx.get("contractAddress")]
    
    erc20_received_values = []
    erc20_sent_values = []
    for tx in erc20_received:
        try:
            decimals = int(tx.get("tokenDecimal", 18))
            value = int(tx.get("value", 0)) / (10 ** decimals)
            erc20_received_values.append(value)
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    
    for tx in erc20_sent:
        try:
            decimals = int(tx.get("tokenDecimal", 18))
            value = int(tx.get("value", 0)) / (10 ** decimals)
            erc20_sent_values.append(value)
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    
    erc20_unique_sent_addr = len(set(tx.get("to", "").lower() for tx in erc20_sent if tx.get("to")))
    erc20_unique_rec_addr = len(set(tx.get("from", "").lower() for tx in erc20_received if tx.get("from")))
    erc20_unique_sent_token = len(set(tx.get("tokenName", "") for tx in erc20_sent))
    erc20_unique_rec_token = len(set(tx.get("tokenName", "") for tx in erc20_received))
    
    # Most common token names
    sent_tokens = [tx.get("tokenName", "") for tx in erc20_sent if tx.get("tokenName")]
    rec_tokens = [tx.get("tokenName", "") for tx in erc20_received if tx.get("tokenName")]
    most_sent_token = max(set(sent_tokens), key=sent_tokens.count) if sent_tokens else ""
    most_rec_token = max(set(rec_tokens), key=rec_tokens.count) if rec_tokens else ""
    
    # Log ether sent (for feature matching)
    log_ether_sent = np.log1p(total_sent * 1e8) if total_sent > 0 else 0.0
    
    # Decile (based on total transactions - rough estimate)
    total_txs = len(sent_txs) + len(received_txs)
    decile = min(9, total_txs // 10)  # Rough approximation
    
    # --- Build feature dictionary ---
    features = {
        "Avg min between sent tnx": avg_time_between(sent_times),
        "Avg min between received tnx": avg_time_between(received_times),
        "Time Diff between first and last (Mins)": time_diff_mins,
        "Sent tnx": len(sent_txs),
        "Received Tnx": len(received_txs),
        "Number of Created Contracts": contract_creates,
        "Unique Received From Addresses": unique_from,
        "Unique Sent To Addresses": unique_to,
        
        # Received value stats
        "min value received": min(received_values) if received_values else 0.0,
        "max value received": max(received_values) if received_values else 0.0,
        "avg val received": np.mean(received_values) if received_values else 0.0,
        
        # Sent value stats
        "min val sent": min(sent_values) if sent_values else 0.0,
        "max val sent": max(sent_values) if sent_values else 0.0,
        "avg val sent": np.mean(sent_values) if sent_values else 0.0,
        
        # Contract value stats
        "min value sent to contract": min(sent_to_contract_values) if sent_to_contract_values else 0.0,
        "max val sent to contract": max(sent_to_contract_values) if sent_to_contract_values else 0.0,
        "avg value sent to contract": np.mean(sent_to_contract_values) if sent_to_contract_values else 0.0,
        
        # Totals
        "total transactions (including tnx to create contract": total_txs + contract_creates,
        "total Ether sent": total_sent,
        "total ether received": total_received,
        "total ether sent contracts": total_sent_contracts,
        "total ether balance": balance,
        
        # ERC20 features
        "Total ERC20 tnxs": len(erc20_txs),
        "ERC20 total Ether received": sum(erc20_received_values),
        "ERC20 total ether sent": sum(erc20_sent_values),
        "ERC20 total Ether sent contract": 0.0,  # Approximation
        "ERC20 uniq sent addr": erc20_unique_sent_addr,
        "ERC20 uniq rec addr": erc20_unique_rec_addr,
        "ERC20 uniq sent addr.1": 0.0,  # Duplicate column in original data
        "ERC20 uniq rec contract addr": len(set(tx.get("contractAddress", "").lower() for tx in erc20_received)),
        
        # ERC20 time features (set to 0 - not critical for model)
        "ERC20 avg time between sent tnx": 0.0,
        "ERC20 avg time between rec tnx": 0.0,
        "ERC20 avg time between rec 2 tnx": 0.0,
        "ERC20 avg time between contract tnx": 0.0,
        
        # ERC20 value stats
        "ERC20 min val rec": min(erc20_received_values) if erc20_received_values else 0.0,
        "ERC20 max val rec": max(erc20_received_values) if erc20_received_values else 0.0,
        "ERC20 avg val rec": np.mean(erc20_received_values) if erc20_received_values else 0.0,
        "ERC20 min val sent": min(erc20_sent_values) if erc20_sent_values else 0.0,
        "ERC20 max val sent": max(erc20_sent_values) if erc20_sent_values else 0.0,
        "ERC20 avg val sent": np.mean(erc20_sent_values) if erc20_sent_values else 0.0,
        "ERC20 min val sent contract": 0.0,
        "ERC20 max val sent contract": 0.0,
        "ERC20 avg val sent contract": 0.0,
        
        "ERC20 uniq sent token name": erc20_unique_sent_token,
        "ERC20 uniq rec token name": erc20_unique_rec_token,
        
        # Derived features
        "decile": decile,
        "log_ether_sent": log_ether_sent,
        
        # Velocity features (set to 0 for live - no historical context yet)
        "tx_count_last_60m": 0.0,
        "ether_sent_last_60m": 0.0,
        "tx_count_last_360m": 0.0,
        "ether_sent_last_360m": 0.0,
        "tx_count_last_1440m": 0.0,
        "ether_sent_last_1440m": 0.0,
        "tx_count_last_10080m": 0.0,
        "ether_sent_last_10080m": 0.0,
        
        # Ratio features
        "ratio_vs_avg_60m": total_sent * 1e8 + 1e-8 if total_sent > 0 else 0.0,
        "ratio_vs_avg_360m": total_sent * 1e8 + 1e-8 if total_sent > 0 else 0.0,
        "ratio_vs_avg_1440m": total_sent * 1e8 + 1e-8 if total_sent > 0 else 0.0,
        "ratio_vs_avg_10080m": total_sent * 1e8 + 1e-8 if total_sent > 0 else 0.0,
    }
    
    return features


def features_to_vector(features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    """Pull the features out into a numpy array in training-column order.
    Missing keys default to 0."""
    return np.array([features.get(name, 0.0) for name in feature_names])


if __name__ == "__main__":
    # quick sanity check with a couple of fake transactions
    mock_data = {
        "address": "0x1234...",
        "normal_transactions": [
            {"from": "0x1234...", "to": "0xabcd", "value": "1000000000000000000", "timeStamp": "1700000000"},
            {"from": "0x5678", "to": "0x1234...", "value": "2000000000000000000", "timeStamp": "1700001000"},
        ],
        "internal_transactions": [],
        "erc20_transfers": [],
        "balance": 1.5,
    }
    
    features = compute_features(mock_data)
    print(f"Computed {len(features)} features")
    print(f"Total Ether sent: {features['total Ether sent']}")
    print(f"Total Ether received: {features['total ether received']}")
