import json
import os
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict, defaultdict
import random
from threading import Lock
from eth_abi import decode
# Configuration
CHAINSTACK_URL = "https://ethereum-mainnet.core.chainstack.com/2b218e28664a4607f447941da0c0518b"
INPUT_FILE = "insertion_dataset_attacker_token_filled_repeated.json"
OUTPUT_FILE = "insertion_dataset_with_extra_profits.json"
CHECKPOINT_FILE = "checkpoint_insertion_dataset.json"
FAILED_FILE = "failed_entries_insertion.log"

ERC20_TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
ERC1155_SINGLE_TRANSFER_TOPIC = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
ERC1155_BATCH_TRANSFER_TOPIC = "0x4a39dc06d4c0dbc64b70af90fd698a233a518aa5d07e595d983b8c0526c8f7fb"
ERC721_TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

MAX_WORKERS = 3
REQUESTS_PER_SECOND = 5
RETRY_LIMIT = 5
MAX_CACHE_SIZE = 20000
BATCH_SIZE = 100

stats = {
    'total_processed': 0,
    'reprocessed': 0,
    'successful_reprocess': 0,
    'failed_reprocess': 0,
    'profit_tx_found': 0,
    'profit_tx_added': 0
}
stats_lock = Lock()
log_lock = Lock()

# Caches
block_cache = OrderedDict()
tx_details_cache = {}

class RateLimiter:
    def __init__(self):
        self.lock = Lock()
        self.last_request = time.time()
        self.min_interval = 1.0 / REQUESTS_PER_SECOND
        
    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_request
            wait_time = max(self.min_interval - elapsed, 0)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request = time.time()

limiter = RateLimiter()

connection_state = {
    'last_success': time.time(),
    'consecutive_fails': 0,
    'max_retries': 100,  
    'backoff_base': 5,   
}
connection_lock = Lock()

def calculate_eta(processed_count, total_count, start_time):
    """Calculate Estimated Time Remaining"""
    if processed_count <= 0:
        return 0
    
    elapsed = time.time() - start_time
    time_per_item = elapsed / processed_count
    remaining_items = total_count - processed_count
    return remaining_items * time_per_item

def check_connection(force_retry=False):
    """Check internet connection with exponential backoff"""
    global connection_state
    
    with connection_lock:
        if connection_state['consecutive_fails'] == 0 and not force_retry:
            return True
            
        backoff = min(
            connection_state['backoff_base'] * 2 ** connection_state['consecutive_fails'],
            3600  
        )
        
        print(f"Connection issues detected. Waiting {backoff:.0f}s (attempt {connection_state['consecutive_fails']+1}/{connection_state['max_retries']})")
        time.sleep(backoff)
        
        test_urls = [
            "https://1.1.1.1",  # Cloudflare DNS
            "https://8.8.8.8",  # Google DNS
            "https://api.chainstack.com"  # Your actual API endpoint
        ]
        
        for url in test_urls:
            try:
                test = requests.head(url, timeout=10)
                if test.status_code < 500:  # Any non-server error is fine
                    connection_state['consecutive_fails'] = 0
                    print("Connection restored! Resuming...")
                    return True
            except:
                continue
        
        # All tests failed
        connection_state['consecutive_fails'] += 1
        if connection_state['consecutive_fails'] >= connection_state['max_retries']:
            raise RuntimeError("Critical network failure after maximum retries")
        return False

def log_failed_entry(entry: Dict, error: str):
    try:
        with log_lock:
            with open(FAILED_FILE, 'a') as f:
                f.write(f"{datetime.now().isoformat()}|{json.dumps(entry)}|{error}\n")
    except Exception as e:
        print(f"Failed to write error log: {str(e)}")

def make_rpc_request(method: str, params: Any) -> Optional[Dict]:
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": random.randint(1, 1000000)
    }

    for attempt in range(RETRY_LIMIT):
        # Check connection before each attempt
        if not check_connection():
            continue
            
        limiter.wait()
        try:
            start_time = time.time()
            response = requests.post(
                CHAINSTACK_URL,
                json=payload,
                timeout=45,
                headers={'Content-Type': 'application/json'}
            )
            response_time = time.time() - start_time
            
            # Update connection state on success
            with connection_lock:
                connection_state['last_success'] = time.time()
                connection_state['consecutive_fails'] = 0
            
            # Log successful requests
            with log_lock:
                with open("api_success_logs.jsonl", "a") as f:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "method": method,
                        "params": params,
                        "status": "success",
                        "response_time": response_time,
                        "attempt": attempt + 1
                    }
                    f.write(json.dumps(log_entry) + "\n")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "params": params,
                "error_type": "HTTPError",
                "status_code": e.response.status_code if e.response else None,
                "response_text": e.response.text[:500] if e.response else None,
                "attempt": attempt + 1,
                "retry": attempt < RETRY_LIMIT - 1
            }
            
            with log_lock:
                with open("api_error_logs.jsonl", "a") as f:
                    f.write(json.dumps(error_data) + "\n")
                
                with open(FAILED_FILE, "a") as f:
                    f.write(f"{datetime.now().isoformat()}|{json.dumps(payload)}|HTTP Error {e.response.status_code if e.response else 'unknown'}: {str(e)}\n")
            
            if e.response and e.response.status_code == 429:
                backoff = min(2 ** attempt, 45) + random.uniform(0, 1)
                print(f"Rate limited, backing off for {backoff:.2f}s (Attempt {attempt+1}/{RETRY_LIMIT})")
                time.sleep(backoff)
                continue
                
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            # Update connection state
            with connection_lock:
                connection_state['consecutive_fails'] += 1
                
            error_type = type(e).__name__
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "params": params,
                "error_type": error_type,
                "attempt": attempt + 1,
                "retry": attempt < RETRY_LIMIT - 1
            }
            
            with log_lock:
                with open("api_error_logs.jsonl", "a") as f:
                    f.write(json.dumps(error_data) + "\n")
                
                with open(FAILED_FILE, "a") as f:
                    f.write(f"{datetime.now().isoformat()}|{json.dumps(payload)}|{error_type}: {str(e)}\n")
            
            # Immediately check connection for persistent issues
            if connection_state['consecutive_fails'] > 2:
                check_connection(force_retry=True)
            continue
            
        except Exception as e:
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "params": params,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "attempt": attempt + 1,
                "retry": False
            }
            
            with log_lock:
                with open("api_error_logs.jsonl", "a") as f:
                    f.write(json.dumps(error_data) + "\n")
                
                with open(FAILED_FILE, "a") as f:
                    f.write(f"{datetime.now().isoformat()}|{json.dumps(payload)}|Unexpected Error ({type(e).__name__}): {e}\n")
            break

    return None

def analyze_error_logs():
    """Analyze error logs to find common issues"""
    error_stats = {
        "total_errors": 0,
        "error_types": defaultdict(int),
        "methods": defaultdict(int),
        "status_codes": defaultdict(int)
    }
    
    try:
        with open("api_error_logs.jsonl", "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    error_stats["total_errors"] += 1
                    error_stats["error_types"][entry.get("error_type", "unknown")] += 1
                    error_stats["methods"][entry.get("method", "unknown")] += 1
                    if "status_code" in entry:
                        error_stats["status_codes"][entry["status_code"]] += 1
                except json.JSONDecodeError:
                    continue
                    
        print("\nAPI Error Analysis:")
        print(f"Total Errors: {error_stats['total_errors']}")
        print("\nError Types:")
        for error_type, count in sorted(error_stats['error_types'].items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
            
        print("\nMethods with Errors:")
        for method, count in sorted(error_stats['methods'].items(), key=lambda x: -x[1]):
            print(f"  {method}: {count}")
            
        print("\nHTTP Status Codes:")
        for code, count in sorted(error_stats['status_codes'].items(), key=lambda x: -x[1]):
            print(f"  {code}: {count}")
            
    except FileNotFoundError:
        print("No error logs found - no errors detected")

def check_api_health():
    test_payload = {
        "jsonrpc": "2.0",
        "method": "web3_clientVersion",
        "params": [],
        "id": 1
    }
    try:
        start_time = time.time()
        response = requests.post(
            CHAINSTACK_URL,
            json=test_payload,
            timeout=10,
            headers={'Content-Type': 'application/json'}
        )
        response_time = time.time() - start_time

        health_status = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response_time,
            "status_code": response.status_code,
            "version": response.json().get("result", "unknown") if response.status_code == 200 else None
        }

        with open("api_health_logs.jsonl", "a") as f:
            f.write(json.dumps(health_status) + "\n")

        return health_status
    except Exception as e:
        error_status = {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e),
            "type": type(e).__name__
        }
        with open("api_health_logs.jsonl", "a") as f:
            f.write(json.dumps(error_status) + "\n")
        return error_status

def hex_to_int(value: Optional[Union[str, int]]) -> int:
    if value is None or value in ("", "0x"):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.startswith("0x"):
            return int(value, 16)
        if value.isdigit():
            return int(value)
    return 0

def get_tx_details(tx_hash: str) -> Optional[Dict]:
    try:
        limiter.wait()
        tx_response = make_rpc_request("eth_getTransactionByHash", [tx_hash])
        if not tx_response or 'result' not in tx_response or not tx_response['result']:
            return None

        tx_data = tx_response['result']
        transfers, success = get_token_transfers(tx_hash)

        return {
            "hash": tx_hash,
            "sender": tx_data.get('from', '').lower(),
            "receiver": tx_data.get('to', '').lower() if tx_data.get('to') else None,
            "transactionIndex": hex_to_int(tx_data.get('transactionIndex')),
            "gasPrice": hex_to_int(tx_data.get('gasPrice')),
            "blockNumber": hex_to_int(tx_data.get('blockNumber')),
            "value": hex_to_int(tx_data.get('value')), 
            "input": tx_data.get('input', ''),
            "nonce": hex_to_int(tx_data.get('nonce')),
            "token_transfers": transfers if success else [],
            "transfer_method": "chainstack"
        }
    except Exception as e:
        log_failed_entry({"tx_hash": tx_hash}, f"get_tx_details error: {str(e)}")
        return None



def find_profit_tx( attacker_address: str, victim_block: int, attack_tokens: List[Dict], victim_tokens: List[Dict], attack_tx_hash: str) -> Optional[str]:
    """Finds profitTx by:
    1. Scanning [victim_block, victim_block+4]
    2. Matching transfers of tokens from attack/victim token_transfers
    3. Ensuring the attacker is the sender (from)
    4. Explicitly excluding attackTx
    5. Now supports ETH, ERC20, ERC777, ERC721 and ERC1155 (single/batch)
    """
    # Extract token addresses from both attack and victim transfers
    token_addresses = set()
    eth_found = False
    
    # Process attack tokens
    for transfer in attack_tokens:
        if transfer.get('tokenType') == 'ETH':
            eth_found = True
            continue
        if 'rawContract' in transfer and 'address' in transfer['rawContract']:
            token_addresses.add(transfer['rawContract']['address'].lower())
    
    # Process victim tokens
    for transfer in victim_tokens:
        if transfer.get('tokenType') == 'ETH':
            eth_found = True
            continue
        if 'rawContract' in transfer and 'address' in transfer['rawContract']:
            token_addresses.add(transfer['rawContract']['address'].lower())
    
    try:
        # Initialize list for candidate transactions
        candidate_txs = []
        
        # 1. Search for token transfers
        if token_addresses:
            # Prepare topics for all token types
            topics = [
                [
                    ERC20_TRANSFER_TOPIC, 
                    ERC721_TRANSFER_TOPIC,
                    ERC1155_SINGLE_TRANSFER_TOPIC,
                    ERC1155_BATCH_TRANSFER_TOPIC
                ],
                None,  # Operator/from placeholder
                None,  # To placeholder
                None   # Additional data
            ]
            
            # Add sender constraint for different token types
            topics[1] = [f"0x{'0'*24}{attacker_address[2:].lower()}"]
            
            params = {
                "fromBlock": hex(victim_block),
                "toBlock": hex(victim_block + 10),
                "topics": topics,
                "address": list(token_addresses)
            }

            logs_response = make_rpc_request("eth_getLogs", [params])
            if logs_response and 'result' in logs_response:
                # Collect candidate transactions
                for log in logs_response['result']:
                    tx_hash = log['transactionHash']
                    if tx_hash.lower() != attack_tx_hash.lower():
                        candidate_txs.append((
                            tx_hash,
                            hex_to_int(log['blockNumber']),
                            hex_to_int(log['transactionIndex'])
                        ))
        
        # 2. Search for ETH transfers (if ETH was involved)
        if eth_found:
            for block_offset in range(0, 5):
                block_num = victim_block + block_offset
                block_response = make_rpc_request(
                    "eth_getBlockByNumber", 
                    [hex(block_num), True]
                )
                
                if block_response and block_response.get('result'):
                    for tx in block_response['result']['transactions']:
                        tx_hash = tx['hash']
                        if (tx_hash.lower() != attack_tx_hash.lower() and
                            tx['from'].lower() == attacker_address.lower() and
                            hex_to_int(tx.get('value', '0x0')) > 0):
                            candidate_txs.append((
                                tx_hash,
                                hex_to_int(tx['blockNumber']),
                                hex_to_int(tx['transactionIndex'])
                            ))
        
        # Find earliest valid TX
        if candidate_txs:
            # Sort by block then transaction index
            candidate_txs.sort(key=lambda x: (x[1], x[2]))
            return candidate_txs[0][0]

        return None

    except Exception as e:
        log_failed_entry({
            "attacker": attacker_address,
            "victim_block": victim_block,
            "token_addresses": list(token_addresses),
            "eth_found": eth_found
        }, f"find_profit_tx error: {str(e)}")
        return None

def get_token_transfers(tx_hash: str) -> (List[Dict], bool):
    """
    Extracts token transfers from transaction receipt
    Now supports: ETH, ERC20, ERC777, ERC721, ERC1155 (single/batch)
    Returns: (transfers, success)
    """
    try:
        receipt_response = make_rpc_request("eth_getTransactionReceipt", [tx_hash])
        if not receipt_response or 'result' not in receipt_response:
            return [], False

        receipt = receipt_response['result']
        if not receipt:
            return [], False

        transfers = []
        logs = receipt.get("logs", [])

        for log in logs:
            topics = log.get("topics", [])
            if not topics:
                continue

            try:
                # ERC20, ERC777, and ERC721 Transfer
                if topics[0].lower() in [ERC20_TRANSFER_TOPIC, ERC721_TRANSFER_TOPIC]:
                    token_type = "ERC20" if topics[0].lower() == ERC20_TRANSFER_TOPIC else "ERC721"
                    transfer_data = {
                        "from": '0x' + topics[1][-40:],
                        "to": '0x' + topics[2][-40:],
                        "rawContract": {"address": log.get("address", "").lower()},
                        "tokenType": token_type
                    }
                    
                    if token_type == "ERC20":
                        transfer_data["value"] = hex_to_int(log.get("data", "0x0"))
                    else:  # ERC721
                        if len(topics) > 3:
                            transfer_data["tokenId"] = hex_to_int(topics[3])
                    
                    transfers.append(transfer_data)

                # ERC1155 Single Transfer
                elif topics[0].lower() == ERC1155_SINGLE_TRANSFER_TOPIC:
                    data = log.get("data", "0x")
                    if len(data) >= 130:
                        token_id = hex_to_int("0x" + data[2:66])
                        value = hex_to_int("0x" + data[66:130])
                        transfers.append({
                            "operator": '0x' + topics[1][-40:],
                            "from": '0x' + topics[2][-40:],
                            "to": '0x' + topics[3][-40:],
                            "tokenId": token_id,
                            "value": value,
                            "rawContract": {"address": log.get("address", "").lower()},
                            "tokenType": "ERC1155"
                        })

                # ERC1155 Batch Transfer
                elif topics[0].lower() == ERC1155_BATCH_TRANSFER_TOPIC:
                    data = log.get("data", "0x")
                    try:
                        # Decode ABI-encoded arrays
                        data_bytes = bytes.fromhex(data[2:])
                        ids_offset = int.from_bytes(data_bytes[:32], 'big')
                        values_offset = int.from_bytes(data_bytes[32:64], 'big')
                        
                        # Decode IDs array
                        ids_length = int.from_bytes(data_bytes[ids_offset:ids_offset+32], 'big')
                        ids = []
                        for i in range(ids_length):
                            start = ids_offset + 32 + i*32
                            ids.append(int.from_bytes(data_bytes[start:start+32], 'big'))
                            
                        # Decode values array
                        values_length = int.from_bytes(data_bytes[values_offset:values_offset+32], 'big')
                        values = []
                        for i in range(values_length):
                            start = values_offset + 32 + i*32
                            values.append(int.from_bytes(data_bytes[start:start+32], 'big'))
                        
                        # Create individual transfers
                        for token_id, amount in zip(ids, values):
                            transfers.append({
                                "operator": '0x' + topics[1][-40:],
                                "from": '0x' + topics[2][-40:],
                                "to": '0x' + topics[3][-40:],
                                "tokenId": token_id,
                                "value": amount,
                                "rawContract": {"address": log.get("address", "").lower()},
                                "tokenType": "ERC1155"
                            })
                    except Exception as e:
                        log_failed_entry({"tx_hash": tx_hash, "log": log}, 
                                        f"ERC1155 batch decode error: {str(e)}")
            except Exception as e:
                log_failed_entry({"tx_hash": tx_hash, "log": log}, 
                                f"Transfer parsing error: {str(e)}")

        return transfers, True

    except Exception as e:
        log_failed_entry({"tx_hash": tx_hash}, f"get_token_transfers error: {str(e)}")
        return [], False

def needs_reprocessing(entry: Dict) -> bool:
    original_data = entry.get('_original', {})
    
    # Only process if:
    # 1. attackTx exists (needed to find profitTx)
    # 2. profits_details is missing or empty
    has_attack = 'attackTx' in original_data and original_data['attackTx']
    missing_profits = (
        'profits_details' not in entry or
        not isinstance(entry['profits_details'], list) or
        len(entry['profits_details']) == 0
    )
    
    return has_attack and missing_profits

def process_entry(entry: Dict) -> Dict:
    if not needs_reprocessing(entry):
        return entry
    
    # Initialize counters with FIXED key names
    entry_stats = {
        'attempted': 0,
        'successful_reprocess': 0,  # Fixed key name
        'failed_reprocess': 0       # Fixed key name
    }
    
    try:
        original_data = entry.get('_original', {})
        attacker_address = entry['attacker_details']['sender']
        
        # Step 1: Find profitTx
        entry_stats['attempted'] = 1  # We're always attempting once per entry
        
        profit_tx_hash = find_profit_tx(
            attacker_address,
            entry['victim_details']['blockNumber'],
            entry['attacker_details'].get('token_transfers', []),
            entry['victim_details'].get('token_transfers', []),
            original_data.get('attackTx')
        )
        
        if profit_tx_hash:
            # Step 2: Fetch token_transfers for profitTx
            tx_details = get_tx_details(profit_tx_hash)
            if tx_details:
                entry['profits_details'] = [tx_details]
                entry['_original']['profitTx'] = profit_tx_hash
                entry_stats['successful_reprocess'] = 1
            else:
                entry_stats['failed_reprocess'] = 1
        else:
            entry_stats['failed_reprocess'] = 1
        
        # Mark as reprocessed
        entry['_reprocessed'] = datetime.now().isoformat()
            
    except Exception as e:
        log_failed_entry(entry, f"Process Entry Error: {e}")
        # If we started processing, count as failed
        entry_stats['failed_reprocess'] = entry_stats.get('attempted', 1)
    
    # Update stats with FIXED key names
    with stats_lock:
        stats['total_processed'] += 1
        stats['reprocessed'] += entry_stats['attempted']
        stats['successful_reprocess'] += entry_stats['successful_reprocess']
        stats['failed_reprocess'] += entry_stats['failed_reprocess']
    
    return entry

def batch_process(entries: List[Dict]) -> List[Dict]:
    # Load checkpoint if exists
    processed = []
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                processed = json.load(f)
            print(f"Loaded checkpoint with {len(processed)} entries")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
    
    # Determine remaining entries to process
    remaining = entries[len(processed):]
    total_entries = len(entries)
    
    # Initialize progress tracking
    start_time = time.time()
    processed_count = len(processed)
    needs_processing = sum(1 for e in remaining if needs_reprocessing(e))
    
    print(f"Processing {needs_processing} of {len(remaining)} remaining entries")
    
    # Add heartbeat timer
    last_heartbeat = time.time()
    HEARTBEAT_INTERVAL = 300  # 5 minutes
    
    # Process in batches
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        batch_count = 0
        
        # Process remaining entries in batches
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i:i+BATCH_SIZE]
            futures.append(executor.submit(
                lambda b: [process_entry(entry) for entry in b], 
                batch
            ))
            batch_count += 1
            
            # Heartbeat check every 10 batches
            if batch_count % 10 == 0:
                current_time = time.time()
                if current_time - last_heartbeat > HEARTBEAT_INTERVAL:
                    print("\nPerforming heartbeat connection check...")
                    check_connection()
                    last_heartbeat = current_time
        
        # Process completed futures
        for i, future in enumerate(as_completed(futures)):
            try:
                batch_result = future.result()
                processed.extend(batch_result)
                processed_count = len(processed)
                
                # Update progress
                elapsed = time.time() - start_time
                progress = processed_count / total_entries
                remaining_entries = total_entries - processed_count
                
                # Calculate ETA
                if elapsed > 0 and processed_count > 0:
                    entries_per_sec = processed_count / elapsed
                    eta = remaining_entries / entries_per_sec if entries_per_sec > 0 else 0
                else:
                    eta = 0
                
                # Print progress
                print(
                f"Processed {processed_count}/{total_entries} "
                    f"({progress:.1%}) | "
                    f"Elapsed: {elapsed/60:.1f}m | "
                    f"ETA: {eta/60:.1f}m | "
                    f"Reprocessed: {stats['reprocessed']} | "
                    f"Success: {stats['successful_reprocess']} | "  # Fixed key
                    f"Failed: {stats['failed_reprocess']}"          # Fixed key
                )
                
                # Save checkpoint every 5 batches
                if i % 5 == 0:
                    with open(CHECKPOINT_FILE, 'w') as f:
                        json.dump(processed, f, indent=2)
                    print(f"Checkpoint saved at {processed_count} entries")
                    
            except Exception as e:
                print(f"Batch processing failed: {str(e)}")
                log_failed_entry(
                    {"batch_index": i, "batch_size": len(batch)},
                    f"Batch processing error: {str(e)}"
                )
    
    # Final checkpoint
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(processed, f, indent=2)
    
    return processed


def main():
    print(f"Starting token transfer processing...")
    start_time = time.time()
    
    # Initial connection check
    print("Verifying network connection...")
    if not check_connection():
        print("No internet connection. Waiting for network...")
        while not check_connection():
            time.sleep(5)
    
    # Initialize stats
    with stats_lock:
        stats.update({
            'total_processed': 0,
            'reprocessed': 0,
            'successful_reprocess': 0,
            'failed_reprocess': 0
        })
    
    # Load data
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        total_entries = len(data)
        print(f"Loaded {total_entries} entries from {INPUT_FILE}")
        
        # Check for checkpoint
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    checkpoint_data = json.load(f)
                if len(checkpoint_data) == total_entries:
                    data = checkpoint_data
                    print(f"Resuming from checkpoint with {len(data)} entries")
                else:
                    print("Checkpoint size mismatch, starting from scratch")
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
        
        # Count entries needing processing
        needs_repair = sum(1 for e in data if needs_reprocessing(e))
        print(f"Found {needs_repair} entries needing profitTx processing")
        
    except Exception as e:
        print(f"Failed to load input: {str(e)}")
        return
    
    # Process data
    processed_data = batch_process(data)
    
    # Verification
    print("\nRunning verification checks...")
    still_needing = sum(1 for e in processed_data if needs_reprocessing(e))
    print(f"Entries still needing processing: {still_needing}")
    
    # Save results
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(processed_data, f, indent=2)
        print(f"\nResults saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nFailed to save results: {str(e)}")
        log_failed_entry({"message": "Final save failed"}, str(e))
    
    # Print summary
    total_time = time.time() - start_time
    print("\nProcessing Summary:")
    print(f"  Entries needing reprocessing: {needs_repair}")
    print(f"  Successfully reprocessed: {stats['successful_reprocess']}")  # Fixed
    print(f"  Failed to reprocess: {stats['failed_reprocess']}")  # Fixed
    print(f"Total processing time: {total_time/60:.1f} minutes")

if __name__ == "__main__":

    main()
