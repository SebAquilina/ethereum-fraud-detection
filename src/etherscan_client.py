"""
Thin Etherscan wrapper. Pulls the normal/internal/ERC20 transaction lists
and the balance for an address. Built for the free tier, so everything goes
through a rate limiter (the free key allows 5 req/s and Etherscan is strict
about it).
"""
import os
import time
import requests
from typing import Optional, Dict, List, Any

# Etherscan API key — must be provided via the ETHERSCAN_API_KEY env var (HF Space secret).
API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
BASE_URL = "https://api.etherscan.io/v2/api"

# Rate limiting for free tier (5 calls/second)
_last_call_time = 0
MIN_INTERVAL = 0.22  # 4.5 calls/second to be safe


def _rate_limit():
    """Enforce rate limiting for free API tier."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_call_time = time.time()


def _api_call(module: str, action: str, max_retries: int = 3, **params) -> Optional[Dict]:
    """Make a rate-limited API call to Etherscan with retries."""
    params.update({
        "chainid": 1,  # Ethereum Mainnet
        "module": module,
        "action": action,
        "apikey": API_KEY,
    })
    
    for attempt in range(max_retries):
        _rate_limit()
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "1" and data.get("message") == "OK":
                return data.get("result")
            elif data.get("message") == "No transactions found":
                return []
            elif "rate limit" in data.get("message", "").lower():
                # Etherscan max rate limit hit, backoff and retry
                backoff = (attempt + 1) * 0.5
                print(f"Rate limit hit. Retrying in {backoff}s...")
                time.sleep(backoff)
                continue
            else:
                print(f"API error: {data.get('message', 'Unknown error')}")
                return None
                
        except requests.Timeout as e:
            print(f"Request timeout on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 1.5)
            else:
                return None
        except requests.RequestException as e:
            print(f"Network error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 1.5)
            else:
                return None
    return None

def get_normal_transactions(address: str, start_block: int = 0, end_block: int = 99999999) -> List[Dict]:
    """Fetch all normal (ETH) transactions for an address."""
    result = _api_call(
        "account", "txlist",
        address=address,
        startblock=start_block,
        endblock=end_block,
        sort="asc"
    )
    return result if result else []


def get_internal_transactions(address: str, start_block: int = 0, end_block: int = 99999999) -> List[Dict]:
    """Fetch all internal transactions for an address."""
    result = _api_call(
        "account", "txlistinternal",
        address=address,
        startblock=start_block,
        endblock=end_block,
        sort="asc"
    )
    return result if result else []


def get_erc20_transfers(address: str, start_block: int = 0, end_block: int = 99999999) -> List[Dict]:
    """Fetch all ERC20 token transfers for an address."""
    result = _api_call(
        "account", "tokentx",
        address=address,
        startblock=start_block,
        endblock=end_block,
        sort="asc"
    )
    return result if result else []


def get_balance(address: str) -> float:
    """Get current ETH balance for an address in Ether."""
    result = _api_call("account", "balance", address=address, tag="latest")
    if result:
        return int(result) / 1e18  # Convert from Wei to Ether
    return 0.0


def get_address_data(address: str) -> Dict[str, Any]:
    """
    Fetch all relevant data for an address.
    Returns a dictionary with transactions, ERC20 transfers, and balance.
    """
    print(f"Fetching data for {address}...")
    
    # Get all transaction types
    normal_txs = get_normal_transactions(address)
    internal_txs = get_internal_transactions(address)
    erc20_txs = get_erc20_transfers(address)
    balance = get_balance(address)
    
    print(f"  Normal TXs: {len(normal_txs)}")
    print(f"  Internal TXs: {len(internal_txs)}")
    print(f"  ERC20 transfers: {len(erc20_txs)}")
    print(f"  Balance: {balance:.4f} ETH")
    
    return {
        "address": address.lower(),
        "normal_transactions": normal_txs,
        "internal_transactions": internal_txs,
        "erc20_transfers": erc20_txs,
        "balance": balance,
    }


# Quick test
if __name__ == "__main__":
    # Test with Vitalik's address
    test_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    data = get_address_data(test_address)
    print(f"\nSuccessfully fetched data. Total normal TXs: {len(data['normal_transactions'])}")
