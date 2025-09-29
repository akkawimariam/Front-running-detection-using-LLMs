import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configuration
ALCHEMY_URL = "https://eth-mainnet.g.alchemy.com/v2/DkqCqVQKFiTWo7O5mGgKq"
INPUT_FILE = "insertion dataset (final).json"
OUTPUT_FILE = "insertion_dataset_alchemy.json"
MAX_WORKERS = 4
REQUESTS_PER_SECOND = 8
BATCH_SIZE = 10
RETRY_LIMIT = 5

class RateLimiter:
    def __init__(self):
        self.last_request = time.time()
        self.backoff = 1.0 / REQUESTS_PER_SECOND
        
    def wait(self):
        elapsed = time.time() - self.last_request
        wait_time = max(self.backoff - elapsed, 0)
        time.sleep(wait_time)
        self.last_request = time.time()

limiter = RateLimiter()

def hex_to_int(hex_str):
    if not hex_str or hex_str == "0x":
        return 0
    if isinstance(hex_str, str) and hex_str.startswith("0x"):
        return int(hex_str, 16)
    elif isinstance(hex_str, str) and hex_str.isdigit():
        return int(hex_str)
    return 0

def make_batch_request(batch_payload):
    for attempt in range(RETRY_LIMIT):
        limiter.wait()
        try:
            response = requests.post(
                ALCHEMY_URL,
                json=batch_payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            if response.status_code == 429:
                time.sleep(5)  # Wait longer on rate limit
                continue
                
            return response.json()
        except Exception:
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

def get_transaction_details(tx_hash):
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getTransactionByHash",
        "params": [tx_hash],
        "id": 1
    }
    
    for attempt in range(RETRY_LIMIT):
        limiter.wait()
        try:
            response = requests.post(ALCHEMY_URL, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if "error" in data:
                return None
                
            tx_data = data.get("result", {})
            if not tx_data:
                return None
                
            # Get transaction receipt for status
            receipt_payload = {
                "jsonrpc": "2.0",
                "method": "eth_getTransactionReceipt",
                "params": [tx_hash],
                "id": 2
            }
            
            receipt_response = requests.post(ALCHEMY_URL, json=receipt_payload, timeout=15)
            receipt_data = receipt_response.json().get("result", {})
            
            # Get block for timestamp
            block_payload = {
                "jsonrpc": "2.0",
                "method": "eth_getBlockByNumber",
                "params": [tx_data.get("blockNumber"), False],
                "id": 3
            }
            
            block_response = requests.post(ALCHEMY_URL, json=block_payload, timeout=15)
            block_data = block_response.json().get("result", {})
            
            return {
                "hash": tx_hash,
                "sender": tx_data.get("from"),
                "receiver": tx_data.get("to"),
                "transactionIndex": hex_to_int(tx_data.get("transactionIndex")),
                "gasPrice": hex_to_int(tx_data.get("gasPrice")),
                "blockNumber": hex_to_int(tx_data.get("blockNumber")),
                "value": hex_to_int(tx_data.get("value")),
                "input": tx_data.get("input"),
                "timestamp": hex_to_int(block_data.get("timestamp", "0x0")),
                "status": hex_to_int(receipt_data.get("status")),
                "nonce": hex_to_int(tx_data.get("nonce")),
                "token_transfers": []  # Empty as per request
            }
            
        except Exception:
            time.sleep(2 ** attempt)
    return None

def process_entry(entry):
    result = {"_original": entry}
    
    # Process attackTx
    if "attackTx" in entry:
        attack_details = get_transaction_details(entry["attackTx"])
        if attack_details:
            result["attacker_details"] = attack_details
    
    # Process victimTx
    if "victimTx" in entry:
        victim_details = get_transaction_details(entry["victimTx"])
        if victim_details:
            result["victim_details"] = victim_details
    
    # Process profitTx (can be single or array)
    if "profitTx" in entry:
        profit_txs = entry["profitTx"]
        if not isinstance(profit_txs, list):
            profit_txs = [profit_txs]
            
        result["profits_details"] = []
        for tx_hash in profit_txs:
            profit_details = get_transaction_details(tx_hash)
            if profit_details:
                result["profits_details"].append(profit_details)
    
    return result

def main():
    print("Starting transaction data retrieval...")
    start_time = time.time()
    
    # Load input data
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries from {INPUT_FILE}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Process entries
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_entry = {executor.submit(process_entry, entry): entry for entry in data}
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_entry)):
            try:
                result = future.result()
                results.append(result)
                
                # Print progress
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {i+1}/{len(data)} entries ({elapsed:.1f}s elapsed)")
                    
            except Exception as e:
                print(f"Error processing entry: {str(e)}")
                # Add original entry even if processing failed
                results.append({"_original": future_to_entry[future]})
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"Processing completed in {total_time:.1f} seconds")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()