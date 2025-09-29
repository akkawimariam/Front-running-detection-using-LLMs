import json
import os
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict, defaultdict

# Configuration
ALCHEMY_URL = "https://eth-mainnet.g.alchemy.com/v2/LRLOUJ-DBuAbt0RPfiHjh"
INPUT_FILE = "dataset_without_heuristi_missing_attackTxns.json"
OUTPUT_FILE = "dataset_without_heuristi_with_attackTxns.json"
CHECKPOINT_FILE = "checkpoint5.json"
FAILED_FILE = "failed_entries5.log"

# Scanning settings
MAX_BLOCKS_BACK = 20  # Always scan 20 blocks maximum
MIN_BLOCKS_BACK = 3   # Always scan at least victim block + 3 previous

# Performance settings
MAX_WORKERS = 3
REQUESTS_PER_SECOND = 10
RETRY_LIMIT = 3
MAX_CACHE_SIZE = 20000
BATCH_SIZE = 100

# Global state
stats = {
    'total_processed': 0,
    'max_blocks_used': 0
}

# Caches
block_cache = OrderedDict()
tx_details_cache = {}

class RateLimiter:
    def __init__(self):
        self.last_request = time.time()
        self.min_interval = 1.0 / REQUESTS_PER_SECOND
        
    def wait(self):
        elapsed = time.time() - self.last_request
        wait_time = max(self.min_interval - elapsed, 0)
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request = time.time()

limiter = RateLimiter()

def log_failed_entry(entry: Dict, error: str):
    try:
        with open(FAILED_FILE, 'a') as f:
            f.write(f"{datetime.now().isoformat()}|{json.dumps(entry)}|{error}\n")
    except Exception as e:
        print(f"Failed to write error log: {str(e)}")

def make_alchemy_request(payload: Any) -> Optional[Dict]:
    for attempt in range(RETRY_LIMIT):
        limiter.wait()
        try:
            response = requests.post(
                ALCHEMY_URL,
                json=payload,
                timeout=20,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                backoff = min(2 ** attempt, 30)
                time.sleep(backoff)
                continue
            log_failed_entry(payload, f"HTTP {e.response.status_code}")
            break
        except Exception as e:
            log_failed_entry(payload, str(e))
            time.sleep(1)
    return None

def batch_request(method: str, params_list: List[Any]) -> List[Optional[Dict]]:
    payload = [{
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": i
    } for i, params in enumerate(params_list)]
    
    response = make_alchemy_request(payload)
    if not response or not isinstance(response, list):
        return [None] * len(params_list)
    
    sorted_results = sorted(response, key=lambda x: x['id'])
    return [item.get('result') for item in sorted_results]

def get_block_cached(block_number: int) -> Optional[Dict]:
    if block_number not in block_cache:
        if len(block_cache) >= MAX_CACHE_SIZE:
            block_cache.popitem(last=False)
        result = batch_request('eth_getBlockByNumber', [[hex(block_number), True]])
        block_cache[block_number] = result[0] if result else None
    return block_cache.get(block_number)

def get_attacker_transactions(attacker: str, start_block: int) -> List[str]:
    # Calculate the mandatory scanning range (victim block + 3 previous)
    mandatory_end = start_block
    mandatory_start = max(0, start_block - MIN_BLOCKS_BACK)
    
    # Calculate the full scanning range (up to 20 blocks back)
    full_start = max(0, start_block - MAX_BLOCKS_BACK)
    
    # First get transactions for the mandatory range
    params = {
        "fromBlock": hex(mandatory_start),
        "toBlock": hex(mandatory_end),
        "fromAddress": attacker,
        "category": ["external"],
        "withMetadata": True
    }
    payload = {
        "jsonrpc": "2.0",
        "method": "alchemy_getAssetTransfers",
        "params": [params],
        "id": 1
    }
    response = make_alchemy_request(payload)
    
    # Group transactions by block
    block_to_txs = defaultdict(list)
    if response and 'result' in response:
        transfers = response['result'].get('transfers', [])
        for t in transfers:
            if t.get('hash') and t.get('blockNum'):
                block_num = hex_to_int(t['blockNum'])
                block_to_txs[block_num].append(t['hash'])
    
    # Now scan beyond mandatory range until we find an empty block
    current_block = mandatory_start - 1
    last_block_with_tx = None
    
    while current_block >= full_start:
        params = {
            "fromBlock": hex(current_block),
            "toBlock": hex(current_block),  # Single block request
            "fromAddress": attacker,
            "category": ["external"],
            "withMetadata": True
        }
        payload = {
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [params],
            "id": 1
        }
        response = make_alchemy_request(payload)
        
        if response and 'result' in response:
            transfers = response['result'].get('transfers', [])
            if transfers:
                for t in transfers:
                    if t.get('hash'):
                        block_to_txs[current_block].append(t['hash'])
                last_block_with_tx = current_block
            else:
                # Found an empty block - stop scanning
                break
        else:
            # API failure - stop scanning
            break
        
        current_block -= 1
    
    # Flatten all transaction hashes
    all_tx_hashes = []
    for block in sorted(block_to_txs.keys(), reverse=True):
        all_tx_hashes.extend(block_to_txs[block])
    
    return list(set(all_tx_hashes))

def get_transaction_batch_details(tx_hashes: List[str]) -> Dict[str, Dict]:
    tx_results = batch_request('eth_getTransactionByHash', [[tx_hash] for tx_hash in tx_hashes])
    tx_details = {tx_hash: result for tx_hash, result in zip(tx_hashes, tx_results)}
    
    receipt_results = batch_request('eth_getTransactionReceipt', [[tx_hash] for tx_hash in tx_hashes])
    receipts = {tx_hash: result for tx_hash, result in zip(tx_hashes, receipt_results)}
    
    trace_hashes = [
        tx_hash for tx_hash in tx_hashes
        if tx_details.get(tx_hash, {}).get('input', '0x') not in ('0x', '', None)
    ]
    trace_results = batch_request('debug_traceTransaction', [
        [tx_hash, {"tracer": "callTracer"}] for tx_hash in trace_hashes
    ])
    traces = {tx_hash: result for tx_hash, result in zip(trace_hashes, trace_results)}
    
    return {
        tx_hash: {
            'tx': tx_details.get(tx_hash),
            'receipt': receipts.get(tx_hash),
            'trace': traces.get(tx_hash)
        }
        for tx_hash in tx_hashes
    }

def hex_to_int(hex_str: Optional[Union[str, int]]) -> int:
    if hex_str is None or hex_str in ("", "0x"):
        return 0
    if isinstance(hex_str, int):
        return hex_str
    if isinstance(hex_str, str):
        if hex_str.startswith("0x"):
            return int(hex_str, 16)
        if hex_str.isdigit():
            return int(hex_str)
    return 0

def scan_backward_for_attacker(attacker: str, start_block: int) -> List[Dict]:
    tx_hashes = get_attacker_transactions(attacker, start_block)
    if not tx_hashes:
        return []
    
    details = get_transaction_batch_details(tx_hashes)
    
    attacker_txs = []
    for tx_hash in tx_hashes:
        detail = details.get(tx_hash, {})
        tx = detail.get('tx')
        receipt = detail.get('receipt')
        trace = detail.get('trace')
        
        if not tx:
            continue
            
        block = get_block_cached(hex_to_int(tx.get('blockNumber')))
        block_num = hex_to_int(tx.get('blockNumber'))
        
        # Track how far back we're finding transactions
        block_diff = start_block - block_num
        stats['max_blocks_used'] = max(stats['max_blocks_used'], block_diff)
        
        attacker_txs.append({
            'hash': tx_hash,
            'from': tx.get('from'),
            'to': tx.get('to'),
            'gasPrice': hex_to_int(tx.get('gasPrice')),
            'timestamp': hex_to_int(block.get('timestamp')) if block else None,
            'blockNumber': block_num,
            'input': tx.get('input', '0x'),
            'nonce': hex_to_int(tx.get('nonce')),
            'transactionIndex': hex_to_int(tx.get('transactionIndex')),
            'value': hex_to_int(tx.get('value')),
            'status': hex_to_int(receipt.get('status')) if receipt else None,
            'error': trace.get('error') if trace else None,
            'opcodes': [call.get('type') for call in trace.get('calls', [])] if trace else None,
            'gas': hex_to_int(tx.get('gas')),
            'gasUsed': hex_to_int(receipt.get('gasUsed')) if receipt else None
        })
    
    return attacker_txs

def process_entry(entry: Dict) -> Dict:
    enriched_entry = entry.copy()
    
    try:
        # Access the nested fields
        #original_data = entry.get('_original', {})
        attacker = entry.get('attacker')
        attacker_details = entry.get('attacker_details', {})

        if not attacker:
            attacker = attacker_details.get('sender')
        victim_details = entry.get('victim_details', {})

        attack_tx = entry.get('attackTx')
        
        # Get victim block from victim_details
        victim_details = entry.get('victim_details', {})
        victim_block = victim_details.get('blockNumber')
        
        if not all([attacker, attack_tx, victim_block]):
            print(f" Missing critical fields: attacker={attacker}, attackTx={attack_tx}, victim_block={victim_block}")
            enriched_entry['attacker_transactions'] = []
            return enriched_entry
        
        # Convert victim block to integer
        victim_block = hex_to_int(victim_block)
        
        # Scan for attacker transactions
        attacker_txs = scan_backward_for_attacker(attacker, victim_block)
        
        # Ensure attackTx is included
        if not any(tx['hash'] == attack_tx for tx in attacker_txs):
            tx_data = batch_request('eth_getTransactionByHash', [[attack_tx]])
            if tx_data and tx_data[0]:
                tx_data = tx_data[0]
                receipt = batch_request('eth_getTransactionReceipt', [[attack_tx]])
                receipt = receipt[0] if receipt else None
                
                trace = None
                if tx_data.get('input', '0x') not in ('0x', '', None):
                    trace = batch_request('debug_traceTransaction', [[attack_tx, {"tracer": "callTracer"}]])
                    trace = trace[0] if trace else None
                
                block = get_block_cached(hex_to_int(tx_data.get('blockNumber')))
                
                attacker_txs.append({
                    'hash': attack_tx,
                    'from': tx_data.get('from'),
                    'to': tx_data.get('to'),
                    'gasPrice': hex_to_int(tx_data.get('gasPrice')),
                    'timestamp': hex_to_int(block.get('timestamp')) if block else None,
                    'blockNumber': hex_to_int(tx_data.get('blockNumber')),
                    'input': tx_data.get('input', '0x'),
                    'nonce': hex_to_int(tx_data.get('nonce')),
                    'transactionIndex': hex_to_int(tx_data.get('transactionIndex')),
                    'value': hex_to_int(tx_data.get('value')),
                    'status': hex_to_int(receipt.get('status')) if receipt else None,
                    'error': trace.get('error') if trace else None,
                    'opcodes': [call.get('type') for call in trace.get('calls', [])] if trace else None,
                    'gas': hex_to_int(tx_data.get('gas')),
                    'gasUsed': hex_to_int(receipt.get('gasUsed')) if receipt else None
                })
        
        enriched_entry['attacker_transactions'] = attacker_txs
        enriched_entry['_processed'] = datetime.now().isoformat()
        
        # Update stats
        stats['total_processed'] += 1
        
    except Exception as e:
        log_failed_entry(entry, str(e))
        enriched_entry['attacker_transactions'] = []
        stats['total_processed'] += 1
    
    return enriched_entry
def has_attacker_transactions(entry: Dict) -> bool:
    """
    Check if entry already has attacker_transactions that are non-empty
    """
    # Check if attacker_transactions exists and is not empty
    if 'attacker_transactions' in entry:
        attacker_txs = entry['attacker_transactions']
        # Check if it's a list with at least one transaction
        if isinstance(attacker_txs, list) and len(attacker_txs) > 0:
            # Additional check: make sure it's not just placeholder data
            if len(attacker_txs) == 1 and attacker_txs[0].get('hash') == 'missing':
                return False
            return True
    return False

def filter_entries_needing_processing(entries: List[Dict]) -> List[Dict]:
    """
    Filter entries that need attacker_transactions processing
    """
    needs_processing = []
    already_has = 0
    
    for entry in entries:
        if has_attacker_transactions(entry):
            already_has += 1
        else:
            needs_processing.append(entry)
    
    print(f"Entries already processed: {already_has}")
    print(f"Entries needing processing: {len(needs_processing)}")
    return needs_processing

def batch_process(entries: List[Dict]) -> List[Dict]:
    processed = []
    start_time = time.time()
    
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                processed = json.load(f)
            print(f"Resuming from checkpoint with {len(processed)} entries")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
    
    # Filter entries that still need processing
    remaining_entries = entries[len(processed):]
    entries_needing_processing = filter_entries_needing_processing(remaining_entries)
    
    total_entries = len(entries_needing_processing)
    
    if total_entries == 0:
        print("No entries need processing!")
        return processed
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i in range(0, len(entries_needing_processing), BATCH_SIZE):
            batch = entries_needing_processing[i:i + BATCH_SIZE]
            futures.append(executor.submit(
                lambda b: [process_entry(e) for e in b], batch
            ))
        
        for i, future in enumerate(as_completed(futures)):
            batch_result = future.result()
            processed.extend(batch_result)
            
            elapsed = time.time() - start_time
            processed_count = len(processed)
            progress = processed_count / total_entries
            eta = (elapsed / processed_count) * (total_entries - processed_count) if processed_count > 0 else 0
            
            print(
                f"Processed {processed_count}/{total_entries} "
                f"({progress:.1%}) | "
                f"Elapsed: {elapsed/60:.1f}m | "
                f"ETA: {eta/60:.1f}m | "
                f"Max Blocks Used: {stats['max_blocks_used']}"
            )
            
            if i % 5 == 0 or i == len(futures) - 1:
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(processed, f)
                print(f"  Checkpoint saved at {processed_count} entries")
    
    return processed

def main():
    print(f"Starting processing (max scan depth: {MAX_BLOCKS_BACK} blocks, min required: {MIN_BLOCKS_BACK+1} blocks)...")
    start_time = time.time()
    
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
        
        total_entries = len(data)
        print(f"Loaded {total_entries} entries from {INPUT_FILE}")

        # Filter entries that need processing
        entries_to_process = filter_entries_needing_processing(data)
        print(f"Entries that need processing: {len(entries_to_process)}/{total_entries}")

    except Exception as e:
        print(f"Failed to load input: {str(e)}")
        return
    
    processed_data = batch_process(entries_to_process)
    
    # Merge processed data with entries that were already complete
    final_data = []
    for entry in data:
        if has_attacker_transactions(entry):
            final_data.append(entry)  # Keep already processed entries
        else:
            # Find the processed version of this entry
            processed_entry = next((e for e in processed_data if e.get('_original', {}).get('hash') == entry.get('hash')), None)
            if processed_entry:
                final_data.append(processed_entry)
            else:
                final_data.append(entry)  # Keep original if not processed
    
    # === YOUR EXISTING VERIFICATION CODE GOES HERE ===
    print("\nRunning verification checks...")
    for i, entry in enumerate(final_data):
        # Access nested fields for verification
        attack_tx = entry.get('attackTx')
        attacker = entry.get('attacker')
        attacker_details = entry.get('attacker_details', {})

        if not attacker:
            # Handle both dict and list formats for attacker_details
            if isinstance(attacker_details, dict):
                attacker = attacker_details.get('sender')
            elif isinstance(attacker_details, list) and len(attacker_details) > 0:
                first_detail = attacker_details[0]
                if isinstance(first_detail, dict):
                    attacker = first_detail.get('sender')
        # Safely get victim_details and handle both dict/list formats
        victim_details = entry.get('victim_details', {})
        if isinstance(victim_details, list) and len(victim_details) > 0:
            victim_details = victim_details[0]  # Take first item if it's a list

        # Now safely access blockNumber
        victim_block = victim_details.get('blockNumber') if isinstance(victim_details, dict) else None
        
        print(f"\nEntry {i+1}:")
        print(f"Attack TX: {attack_tx}")
        print(f"Attacker: {attacker}")
        
        if victim_block is None:
            print("ERROR: No victim block number found")
            continue
            
        victim_block = hex_to_int(victim_block)
        print(f"Victim Block: {victim_block}")
        
        attacker_txs = entry.get('attacker_transactions', [])
        print(f"Found {len(attacker_txs)} attacker transactions")
        
        for tx in attacker_txs:
            try:
                assert tx['from'].lower() == attacker.lower(), \
                    f"Sender {tx['from']} != attacker {attacker}"
                assert tx['blockNumber'] <= victim_block, \
                    f"Transaction at block {tx['blockNumber']} > victim block {victim_block}"
                print(f" Valid tx {tx['hash'][:10]}... (Block {tx['blockNumber']})")
            except AssertionError as e:
                print(f" Invalid tx {tx['hash'][:10]}...: {str(e)}")
            except Exception as e:
                print(f" Error verifying tx: {str(e)}")
    # === END OF VERIFICATION CODE ===
    
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(final_data, f, indent=2)  
        print(f"\nResults saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nFailed to save results: {str(e)}")
        log_failed_entry({"message": "Final save failed"}, str(e))
    
    total_time = time.time() - start_time
    
    print(f"\nMax blocks used in any attack: {stats['max_blocks_used']}")
    print(f"Total processing time: {total_time/60:.1f} minutes")

main()