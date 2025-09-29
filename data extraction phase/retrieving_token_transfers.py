import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
import traceback

# Configuration - Optimized for Alchemy
ALCHEMY_URL = "https://eth-mainnet.g.alchemy.com/v2/DkqCqVQKFiTWo7O5mGgKq"
INPUT_FILE = "insertion dataset (final).json"
OUTPUT_FILE = "insertion_dataset_alchemy.json"
CHECKPOINT_FILE = "insertion_checkpoint_alchemy.json"
FAILED_TXS_FILE = "failed_transactions_alchemy.json"

# Optimized settings for Alchemy
MAX_WORKERS = 4
REQUESTS_PER_SECOND = 8
BATCH_SIZE = 10
RETRY_LIMIT = 5
ENTRY_BATCH_SIZE = 10

class RateLimiter:
    def __init__(self):
        self.last_request = time.time()
        self.backoff = 1.0 / REQUESTS_PER_SECOND
        self.request_count = 0
        
    def wait(self):
        elapsed = time.time() - self.last_request
        wait_time = max(self.backoff - elapsed, 0)
        time.sleep(wait_time)
        self.last_request = time.time()

limiter = RateLimiter()

def safe_json_load(filepath):
    try:
        with open(filepath) as f:
            return json.load(f) 
    except json.JSONDecodeError:
        data = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data

def atomic_write(data, filename):
    temp = f"{filename}.tmp"
    with open(temp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(temp, filename)

def hex_to_int(hex_str):
    if not hex_str or hex_str == "0x":
        return 0
    if isinstance(hex_str, str) and hex_str.startswith("0x"):
        return int(hex_str, 16)
    elif isinstance(hex_str, str) and hex_str.isdigit():
        return int(hex_str)
    else:
        raise ValueError(f"Cannot convert {hex_str} to integer")

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
                retry_after = int(response.headers.get('Retry-After', 5))
                limiter.backoff = retry_after * 1.5
                time.sleep(limiter.backoff)
                continue
                
            limiter.backoff = 1.0 / REQUESTS_PER_SECOND
            return response.json()
        except Exception as e:
            sleep_time = (2 ** attempt) + random.uniform(0, 1)  # Add jitter
            log_error(f"Batch attempt {attempt+1} failed: {str(e)}")
            #time.sleep(2 ** attempt)
            time.sleep(sleep_time)
    return None

def log_error(message):
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] {message}\n"
    with open(FAILED_TXS_FILE, 'a') as f:
        f.write(log_entry)


def get_asset_transfers(tx_hash: str) -> (List[Dict], bool):
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

def get_receipts_batch(tx_hashes):
    if not tx_hashes:
        return {}
        
    batch = []
    id_to_txhash = {}
    for i, tx_hash in enumerate(tx_hashes):
        request_id = f"receipt_{i}"
        batch.append({
            "jsonrpc": "2.0",
            "method": "eth_getTransactionReceipt",
            "params": [tx_hash],
            "id": request_id
        })
        id_to_txhash[request_id] = tx_hash
    
    response = make_batch_request(batch)
    if not response:
        return {}
    
    receipts = {}
    for item in response:
        if "result" in item:
            request_id = item["id"]
            tx_hash = id_to_txhash.get(request_id)
            if tx_hash:
                receipts[tx_hash] = item["result"]
    return receipts

def get_blocks_batch(block_numbers):
    if not block_numbers:
        return {}
    
    batch = []
    id_to_blocknum = {}
    for i, block_num in enumerate(block_numbers):
        request_id = f"block_{i}"
        batch.append({
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": [block_num, False],
            "id": request_id
        })
        id_to_blocknum[request_id] = block_num
    
    response = make_batch_request(batch)
    if not response:
        return {}
    
    blocks = {}
    for item in response:
        if "result" in item:
            request_id = item["id"]
            block_num = id_to_blocknum.get(request_id)
            if block_num:
                blocks[block_num] = item["result"]
    return blocks

def print_progress(current, total, start_time):
    elapsed = time.time() - start_time
    percent = (current / total) * 100
    eta = (elapsed / current) * (total - current) if current > 0 else 0
    print(
        f"Processed: {current}/{total} ({percent:.1f}%) | "
        f"Elapsed: {elapsed/60:.1f}m | "
        f"ETA: {eta/60:.1f}m",
        end='\r'
    )


def process_batch(entries):
    batch_payload = []
    tx_map = {}
    all_tx_hashes = set()
    block_numbers = set()
    
    for idx, entry in enumerate(entries):
        entry_id = str(idx)
        
        # Preserve original entry structure
        result_entry = {
            "_original": entry,  # Keep original intact
            "_processed": {
                "success": True,  # Default, will update later
                "errors": [],
                "processed_at": datetime.now().isoformat()
            }
        }
        
        if "victimTx" in entry:
            tx_hash = entry["victimTx"]
            request_id = f"{entry_id}_victim"
            batch_payload.append({
                "jsonrpc": "2.0",
                "method": "eth_getTransactionByHash",
                "params": [tx_hash],
                "id": request_id
            })
            tx_map[request_id] = (idx, "victim", tx_hash)
            all_tx_hashes.add(tx_hash)
        
        if "attackTx" in entry:
            tx_hash = entry["attackTx"]
            request_id = f"{entry_id}_attacker"
            batch_payload.append({
                "jsonrpc": "2.0",
                "method": "eth_getTransactionByHash",
                "params": [tx_hash],
                "id": request_id
            })
            tx_map[request_id] = (idx, "attacker", tx_hash)
            all_tx_hashes.add(tx_hash)
        
        if "profitTx" in entry:
            profit_txs = entry["profitTx"]
            if not isinstance(profit_txs, list):
                profit_txs = [profit_txs]  # Ensure it's always a list
                
            for profit_idx, tx_hash in enumerate(profit_txs):
                request_id = f"{entry_id}_profit_{profit_idx}"
                batch_payload.append({
                    "jsonrpc": "2.0",
                    "method": "eth_getTransactionByHash",
                    "params": [tx_hash],
                    "id": request_id
                })
                tx_map[request_id] = (idx, "profit", tx_hash, profit_idx)
                all_tx_hashes.add(tx_hash)

    tx_response = make_batch_request(batch_payload)
    if not tx_response:
        return create_failed_entries(entries)
    
    receipts = get_receipts_batch(list(all_tx_hashes))
    
    for response_item in tx_response:
        if "result" in response_item:
            tx_data = response_item["result"]
            if tx_data and "blockNumber" in tx_data:
                block_numbers.add(tx_data["blockNumber"])
    
    blocks = get_blocks_batch(list(block_numbers))
    
    # Initialize results with original data preserved
    results = []
    for idx, entry in enumerate(entries):
        results.append({
            "_original": entry,
            "_processed": {
                "success": True,  # Temporary
                "errors": [],
                "processed_at": datetime.now().isoformat()
            }
        })
    
    # Get asset transfers
    asset_transfers_map = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_tx = {executor.submit(get_asset_transfers, tx_hash): tx_hash for tx_hash in all_tx_hashes}
        for future in as_completed(future_to_tx):
            tx_hash = future_to_tx[future]
            try:
                asset_transfers_map[tx_hash] = future.result()
            except Exception as e:
                log_error(f"Failed to get transfers for {tx_hash}: {str(e)}")
                asset_transfers_map[tx_hash] = []
    
    # Process transaction responses
    for response_item in tx_response:
        request_id = str(response_item.get("id", ""))
        
        if "error" in response_item or request_id not in tx_map:
            continue
            
        tx_data = response_item.get("result", {})
        tx_hash = tx_data.get("hash", "")
        receipt = receipts.get(tx_hash, {})
        block_data = blocks.get(tx_data.get("blockNumber", ""), {})
        
        token_transfers = asset_transfers_map.get(tx_hash, [])
        
        # Build transaction result with token_transfers
        tx_result = {
            "hash": tx_hash,
            "sender": tx_data.get("from"),
            "receiver": tx_data.get("to"),
            "transactionIndex": hex_to_int(tx_data.get("transactionIndex")),
            "gasPrice": hex_to_int(tx_data.get("gasPrice")),
            "blockNumber": hex_to_int(tx_data.get("blockNumber")),
            "value": hex_to_int(tx_data.get("value")),
            "input": tx_data.get("input"),
            "timestamp": hex_to_int(block_data.get("timestamp", "0x0")) if block_data else 0,
            "status": hex_to_int(receipt.get("status")),
            "nonce": hex_to_int(tx_data.get("nonce")),
            "token_transfers": token_transfers,
            "_status": "success" if tx_data else "failed"
        }
        
        idx, tx_type, *extra = tx_map[request_id]
        
        # Add to appropriate field in results
        if tx_type == "victim":
            results[idx]["victim_details"] = tx_result
        elif tx_type == "attacker":
            results[idx]["attacker_details"] = tx_result
        elif tx_type == "profit":
            if "profits_details" not in results[idx]:
                results[idx]["profits_details"] = []
                
            profit_idx = extra[1] if len(extra) > 1 else 0
            # Ensure proper indexing
            while len(results[idx]["profits_details"]) <= profit_idx:
                results[idx]["profits_details"].append({})
            results[idx]["profits_details"][profit_idx] = tx_result
    
    # Final error checking
    for result in results:
        errors = []
        
        if "victim_details" in result and result["victim_details"].get("_status") != "success":
            errors.append("victimTx failed")
        
        if "attacker_details" in result and result["attacker_details"].get("_status") != "success":
            errors.append("attackTx failed")
        
        if "profits_details" in result:
            for i, profit in enumerate(result["profits_details"]):
                if profit.get("_status") != "success":
                    errors.append(f"profitTx {i} failed")
        
        # Update processing status
        result["_processed"] = {
            "success": len(errors) == 0,
            "errors": errors,
            "processed_at": datetime.now().isoformat()
        }
    
    return results

def create_failed_entries(entries):
    results = []
    for entry in entries:
        result = {
            "_original": entry,
            "_processed": {
                "success": False,
                "errors": ["Batch request failed"],
                "processed_at": datetime.now().isoformat()
            }
        }
        
        # Ensure token_transfers exists even for failed transactions
        if "victimTx" in entry:
            result["victim_details"] = {
                "_status": "failed",
                "hash": entry["victimTx"],
                "token_transfers": []
            }
            
        if "attackTx" in entry:
            result["attacker_details"] = {
                "_status": "failed", 
                "hash": entry["attackTx"],
                "token_transfers": []
            }
            
        if "profitTx" in entry:
            txs = entry["profitTx"] if isinstance(entry["profitTx"], list) else [entry["profitTx"]]
            result["profits_details"] = [{
                "_status": "failed",
                "hash": tx,
                "token_transfers": []
            } for tx in txs]
            
        results.append(result)
    return results

def save_checkpoint(results):
    try:
        # Simply save the full results list
        temp = f"{CHECKPOINT_FILE}.tmp"
        with open(temp, 'w') as f:
            json.dump(results, f, indent=2)
        os.replace(temp, CHECKPOINT_FILE)
        print(f"Checkpoint saved with {len(results)} entries")
    except Exception as e:
        log_error(f"Checkpoint save failed: {str(e)}")

def append_to_output(batch_results, output_file):
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump([], f, indent=2)
    
    with open(output_file, 'r') as f:
        existing_data = json.load(f)
    
    existing_data.extend(batch_results)
    
    temp_file = f"{output_file}.tmp"
    with open(temp_file, 'w') as f:
        json.dump(existing_data, f, indent=2)
    os.replace(temp_file, output_file)

def main():
    print("Starting optimized processing with Alchemy API...")
    start_time = time.time()
    
    try:
        data = safe_json_load(INPUT_FILE)
        total_entries = len(data)
        print(f"Loaded {total_entries} entries from {INPUT_FILE}")
    except Exception as e:
        log_error(f"Data load failed: {str(e)}")
        print(f"Error loading data: {str(e)}")
        return
    
    processed = []
    if os.path.exists(CHECKPOINT_FILE):
        try:
            processed = safe_json_load(CHECKPOINT_FILE)
            print(f"Resuming from checkpoint: {len(processed)} entries processed")
        except:
            print("Checkpoint found but could not load, starting fresh")
    
    remaining = data[len(processed):]
    total_batches = (len(remaining) + ENTRY_BATCH_SIZE - 1) // ENTRY_BATCH_SIZE
    print(f"Processing {len(remaining)} remaining entries in {total_batches} batches...")
    
    CHECKPOINT_INTERVAL = 3000 // ENTRY_BATCH_SIZE
    last_checkpoint = 0
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            completed_count = 0
            
            for batch_idx in range(0, len(remaining), ENTRY_BATCH_SIZE):
                batch = remaining[batch_idx:batch_idx + ENTRY_BATCH_SIZE]
                futures.append(executor.submit(process_batch, batch))
            
            for future in as_completed(futures):
                completed_count += 1
                print_progress(completed_count, total_batches, start_time)
                
                try:
                    batch_results = future.result()
                    processed.extend(batch_results)
                    
                    if (completed_count % CHECKPOINT_INTERVAL == 0) or (completed_count == total_batches):
                        entries_since_last_checkpoint = len(processed) - last_checkpoint
                        if entries_since_last_checkpoint > 0:
                            save_checkpoint(processed)
                            print(f"\nCheckpoint saved at {len(processed)} entries")
                            last_checkpoint = len(processed)
                        
                except Exception as e:
                    log_error(f"Batch processing failed: {str(e)}\n{traceback.format_exc()}")
                    print(f"\nBatch {completed_count} failed: {str(e)}")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving checkpoint...")
        save_checkpoint(processed)
        return
    
    try:
        if len(processed) > last_checkpoint:
            save_checkpoint(processed)
            print(f"\nFinal checkpoint saved at {len(processed)} entries")
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in processed if r["_processed"]["success"])
        
        print(f"\nProcessing completed in {total_time/3600:.2f} hours")
        print(f"Results saved to {CHECKPOINT_FILE}")
        print(f"Success rate: {success_count}/{len(processed)} ({success_count/len(processed):.1%})")
        
    except Exception as e:
        print(f"\nFailed to save results: {str(e)}")
        log_error(f"Result save failed: {str(e)}")


if __name__ == "__main__":
    main()