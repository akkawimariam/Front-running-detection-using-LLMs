#!/usr/bin/env python3
"""
Frontrunning Attack Classification - Converting Causal LM to Classification
Input Length Ablation Study: 64, 128, 256, 512 tokens.
With LoRA and 4-bit Quantization
"""

import os
import logging
import json
import math
from datetime import datetime
import warnings
import gc
import random
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from torch import nn
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import Dataset
import torch.nn.functional as F

# Suppress unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ================ GLOBAL CONFIG ================
DEFAULT_MAX_LENGTH = 128
NUM_LABELS = 3  # displacement, insertion, suppression

# ================ MODELS CONFIGURATION ================
MODEL_CONFIGS = {
    "gemma2": {
        "name": "Gemma-2-2B-IT",
        "path": "./gemma-2-2b-it",
        "torch_dtype": "float16",  
        "batch_size": 4,
        "lora_rank": 16,
        "lora_alpha": 32,
    },
    "llama3": {
        "name": "Llama-3.2-3B-Instruct", 
        "path": "./llama-3.2-3b-instruct",
        "torch_dtype": "float16",  
        "batch_size": 2,
        "lora_rank": 16,
        "lora_alpha": 32,
    }
}

# ================ QUANTIZATION & LORA CONFIG ================
BITSANDBYTES_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

DEFAULT_LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# ================ INPUT LENGTH EXPERIMENT CONFIG ================
INPUT_LENGTH_EXPERIMENTS = [64, 128, 256]

# ================ HELPER FUNCTIONS ================
def setup_logging(output_dir):
    """Sets up professional logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def build_input_text(entry, fixed_seed=42):
    """Build a model input string from a cluster of transactions."""
    cluster = entry.get('cluster', [])
    if not cluster:
        return ""

    cluster_str = str([tx.get('hash', '') for tx in cluster])
    cluster_hash = hash(cluster_str) % 10000
    local_random = random.Random(fixed_seed + cluster_hash)
    local_random.shuffle(cluster)

    txn_strs = []
    for txn in cluster:
        gas_price = txn.get('gasPrice', 0)
        gas_used  = txn.get('gasUsed', 0)
        gas_limit = txn.get('gas', txn.get('gasLimit', 0))  
        timestamp = txn.get('timestamp', 0)
        blockNumber = txn.get('blockNumber', 0)
        transactionIndex = txn.get('transactionIndex', 0)
        tx_input = txn.get('input', '')
        nonce = txn.get('nonce', 0)
        sender   = txn.get('sender')   or txn.get('from')   or "null"
        receiver = txn.get('receiver') or txn.get('to')     or "null"

        if tx_input.startswith("0x") and len(tx_input) >= 10:
            function_selector = tx_input[2:10]
        else:
            function_selector = "null"

        txn_str = (
            f"sender:{sender} | receiver:{receiver} | gasPrice:{int(gas_price)} | "
            f"gasUsed:{int(gas_used)} | gasLimit:{int(gas_limit)} | "
            f"timestamp:{timestamp} | blockNumber:{blockNumber} | "
            f"transactionIndex:{transactionIndex} | nonce:{nonce} | "
            f"input_truncated:{function_selector}"
        )
        txn_strs.append(txn_str)

    input_str = " || ".join(txn_strs)
    return input_str

def format_for_classification(text, label):
    """Format text for classification model with instruction."""
    instruction = (
        "Classify this transaction cluster's frontrunning attack type. "
        "Options: displacement, insertion, suppression.\n\n"
        "Transactions:\n"
    )
    
    prompt = f"{instruction}{text}"
    return prompt

def compute_metrics(eval_pred):
    """Compute metrics for classification task."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    classes = [0, 1, 2]
    class_names = ["displacement", "insertion", "suppression"]
    
    results = {'accuracy': accuracy}
    
    for cls in classes:
        tp = np.sum((predictions == cls) & (labels == cls))
        fp = np.sum((predictions == cls) & (labels != cls))
        fn = np.sum((predictions != cls) & (labels == cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[f'{class_names[cls]}_precision'] = precision
        results[f'{class_names[cls]}_recall'] = recall
        results[f'{class_names[cls]}_f1'] = f1
    
    macro_precision = np.mean([results[f'{name}_precision'] for name in class_names])
    macro_recall = np.mean([results[f'{name}_recall'] for name in class_names])
    macro_f1 = np.mean([results[f'{name}_f1'] for name in class_names])
    
    results.update({
        'macro_precision': macro_precision,
        'macro_recall': macro_recall, 
        'macro_f1': macro_f1
    })
    
    return results

# ================ DATA & TRAINING CLASSES ================
class LossCallback(TrainerCallback):
    """Callback to save loss history for plotting."""
    def __init__(self):
        self.losses = []
        self.steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            self.steps.append(state.global_step)
    
    def save_loss_data(self, output_path):
        """Save loss data as JSON for later analysis"""
        loss_data = {
            'steps': self.steps,
            'losses': self.losses,
            'metadata': {
                'min_loss': min(self.losses) if self.losses else None,
                'final_loss': self.losses[-1] if self.losses else None,
                'total_steps': len(self.steps)
            }
        }
        with open(output_path, 'w') as f:
            json.dump(loss_data, f, indent=4)

class FrontrunningDataset:
    """Manages dataset loading and tokenization for classification."""
    def __init__(self, model_path, max_length):
        self.logger = logging.getLogger(__name__)
        self.max_length = max_length
        self.tokenizer = self._load_tokenizer(model_path)
        self.data = {}

    def _load_tokenizer(self, model_path):
        """Load and configure the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.logger.info(f"Tokenizer pad_token set to {tokenizer.pad_token}")
        return tokenizer

    def load_data(self, data_path, split="train"):
        """Load dataset from JSON and return HF Dataset object."""
        file_path = os.path.join(data_path, f"{split}.json")
        self.logger.info(f"Loading data from {file_path}")
        
        with open(file_path, 'r') as f:
            data_list = json.load(f)

        if not data_list:
            raise ValueError(f"No data found in {file_path}")

        labels = [entry.get("label", 0) for entry in data_list]
        texts = [build_input_text(entry) for entry in data_list]
        
        formatted_texts = [format_for_classification(text, label) for text, label in zip(texts, labels)]
        
        self.data[split] = Dataset.from_dict({"text": formatted_texts, "labels": labels})
        return self.data[split]

    def tokenize_dataset(self, split):
        """Tokenize a specific split of the dataset with caching."""
        dataset = self.data[split]
        os.makedirs(".cache", exist_ok=True)
        cache_file = f".cache/tokenized_{split}_{self.max_length}.arrow"

        if os.path.exists(cache_file):
            try:
                return Dataset.load_from_disk(cache_file)
            except Exception:
                os.remove(cache_file)

        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )
            tokenized["labels"] = examples["labels"]
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, batch_size=500,
            remove_columns=dataset.column_names, load_from_cache_file=False
        )

        try:
            tokenized_dataset.save_to_disk(cache_file)
        except Exception as e:
            self.logger.warning(f"Could not save cache: {e}")

        return tokenized_dataset

class ResearchTrainer:
    """Manages the end-to-end training and evaluation process."""
    
    def __init__(self, args, max_length, model_config, model_key, use_lora=True, use_quantization=True):
        self.args = args
        self.max_length = max_length
        self.model_config = model_config
        self.model_key = model_key
        self.use_lora = use_lora
        self.use_quantization = use_quantization
        self.logger = setup_logging(args.output_dir)
        
        method_suffix = ""
        if use_lora:
            method_suffix += "_lora"
        if use_quantization:
            method_suffix += "_4bit"
        
        self.output_dir = os.path.join(
            args.output_dir,
            f"{model_key}_len{max_length}{method_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        set_seed(args.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Training with LoRA: {use_lora}, Quantization: {use_quantization}")
        self.logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.dataset_manager = FrontrunningDataset(model_config["path"], max_length)
        if self.dataset_manager.tokenizer.pad_token is None:
            self.dataset_manager.tokenizer.pad_token = self.dataset_manager.tokenizer.eos_token
            self.dataset_manager.tokenizer.cls_token = self.dataset_manager.tokenizer.eos_token

        self.model = self._load_model()

    def _load_model(self):
        """Load model with quantization and LoRA optimizations."""
        torch.cuda.empty_cache()
        
        self.logger.info(f"Loading base model: {self.model_config['path']}")
        self.logger.info("Converting causal LM model to sequence classification...")
        
        try:
            torch_dtype = getattr(torch, self.model_config['torch_dtype'])
            
            quantization_config = BITSANDBYTES_CONFIG if self.use_quantization else None
            
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config["path"],
                num_labels=NUM_LABELS,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
            )
            model.config.pad_token_id = self.dataset_manager.tokenizer.pad_token_id 
            model.resize_token_embeddings(len(self.dataset_manager.tokenizer))
            

            if hasattr(model, "score"):
                torch.nn.init.normal_(model.score.weight, std=0.02)
                if model.score.bias is not None:
                    torch.nn.init.zeros_(model.score.bias)

            if self.use_quantization:
                model = prepare_model_for_kbit_training(model)
            
            if self.use_lora:
                lora_config = self._get_lora_config()
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
            
            self.logger.info("Model loaded successfully with optimizations")
            return model
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.error("GPU out of memory! Trying CPU mode...")
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_config["path"],
                    num_labels=NUM_LABELS,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map=None,
                )
                return model.to('cpu')
            else:
                raise e

    def _get_lora_config(self):
        """Get LoRA configuration with model-specific parameters."""
        lora_r = self.model_config.get("lora_rank", DEFAULT_LORA_CONFIG.r)
        lora_alpha = self.model_config.get("lora_alpha", DEFAULT_LORA_CONFIG.lora_alpha)
        
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    def calculate_safe_weights(self, class_counts):
        """Calculate class weights for imbalanced data."""
        smoothing_factor = 0.5
        adjusted_counts = class_counts + smoothing_factor
        weights = 1.0 / adjusted_counts
        weights = weights / weights.sum() * len(class_counts)
        self.logger.info(f"Class weights: {weights}")
        return torch.tensor(weights, dtype=torch.float32)

    def train(self):
        """Main training loop for classification."""
        self.logger.info(f"Starting training with max_length={self.max_length}")

        train_dataset = self.dataset_manager.load_data(self.args.dataset_path, "train")
        val_dataset = self.dataset_manager.load_data(self.args.dataset_path, "validation")
        test_dataset = self.dataset_manager.load_data(self.args.dataset_path, "test")

        tokenized_train = self.dataset_manager.tokenize_dataset("train")
        tokenized_val = self.dataset_manager.tokenize_dataset("validation")
        tokenized_test = self.dataset_manager.tokenize_dataset("test")

        class_counts = np.bincount(train_dataset['labels'], minlength=3)
        class_weights = self.calculate_safe_weights(class_counts)
        if self.device.type == "cuda":
            class_weights = class_weights.to(self.device)
            
        self.logger.info(f"Class counts: {class_counts}, weights: {class_weights}")

        batch_size = self.model_config.get("batch_size", self.args.batch_size)
        if self.max_length == 256:  
            batch_size = max(1, batch_size // 2) 
            self.logger.info(f"Reduced batch size to {batch_size} for 256-token sequence")
    
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, "temp_training"),
            overwrite_output_dir=True,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=self.args.grad_accum_steps,
            learning_rate=self.args.lr,
            optim="adamw_torch",
            weight_decay=0.1,
            fp16=True,
            dataloader_num_workers=0,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=True,
            report_to="none",
            logging_steps=10,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            max_grad_norm=1.0,
            gradient_checkpointing=True,
            disable_tqdm=False,  
        )

        loss_callback = LossCallback()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.dataset_manager.tokenizer,
            data_collator=DataCollatorWithPadding(
                tokenizer=self.dataset_manager.tokenizer,
                padding=True,
                max_length=self.max_length
            ),
            compute_metrics=compute_metrics,
            callbacks=[loss_callback, EarlyStoppingCallback(early_stopping_patience=2)],
        )

        self.logger.info("*** Starting Classification Training ***")
        
        try:
            train_result = trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.logger.error("Training failed due to GPU memory issues")
                return {
                    "model": self.model_key,
                    "max_length": self.max_length,
                    "error": "GPU out of memory",
                    "test_metrics": {"eval_accuracy": 0.0, "eval_macro_f1": 0.0}
                }
            else:
                raise e

        test_metrics = None
        evaluation_error = None

        try:
            self.logger.info("*** Evaluating on Test Set ***")
            test_metrics = trainer.evaluate(tokenized_test)
            self.logger.info(f"Test metrics: {test_metrics}")
        except Exception as e:
            evaluation_error = str(e)
            self.logger.error(f"Evaluation failed: {e}")
            test_metrics = {'eval_accuracy': 0.0, 'eval_macro_f1': 0.0, 'error': evaluation_error}

        # Clean all data before saving
        results = {
            "model": self.model_key,
            "max_length": self.max_length,
            "use_lora": self.use_lora,
            "use_quantization": self.use_quantization,
            "training_stats": self.clean_json(train_result.metrics) if 'train_result' in locals() else {},
            "test_metrics": self.clean_json(test_metrics),
            "evaluation_error": evaluation_error
        }
        
        self._save_results(results, training_args, loss_callback)
        
        try:
            if self.use_lora:
                trainer.model.save_pretrained(os.path.join(self.output_dir, "lora_adapters"))
            else:
                trainer.save_model(os.path.join(self.output_dir, "final_model"))
            
            self.dataset_manager.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))
        except Exception as e:
            self.logger.warning(f"Could not save model: {e}")
        
        return results
 
    def clean_json(self, obj):
        """Convert non-serializable objects to serializable ones."""
        if obj is None:
            return None
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        elif isinstance(obj, (torch.dtype, torch.device)):
            return str(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self.clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [self.clean_json(x) for x in obj]
        elif hasattr(obj, '__dict__'):
            return self.clean_json(obj.__dict__)
        else:
            try:
                return str(obj)
            except:
                return f"<unserializable object: {type(obj).__name__}>"

    def _save_results(self, results, training_args, loss_callback):
        """Saves all results and metadata to a JSON file."""
        
        def nuclear_clean(obj):
            """Convert EVERYTHING to JSON-serializable types."""
            if obj is None:
                return None
            elif isinstance(obj, (int, float, str, bool)):
                return obj
            elif isinstance(obj, (torch.dtype, torch.device)):
                return str(obj)
            elif isinstance(obj, torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {str(k): nuclear_clean(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return [nuclear_clean(x) for x in obj]
            elif hasattr(obj, '__dict__'):
                return nuclear_clean(obj.__dict__)
            else:
                try:
                    return str(obj)
                except:
                    return f"<unserializable: {type(obj).__name__}>"
        
        final_output_path = os.path.join(self.output_dir, "final_results.json")
        training_args_dict = training_args.to_dict()
        training_args_dict.pop("report_to", None)
        
        full_results = {
            "args": nuclear_clean(vars(self.args)),
            "model_config": nuclear_clean(self.model_config),
            "training_args": nuclear_clean(training_args_dict),
            "optimization_methods": nuclear_clean({
                "lora": self.use_lora,
                "quantization": self.use_quantization
            }),
            "metrics": nuclear_clean(results)
        }
        
        with open(final_output_path, 'w') as f:
            json.dump(full_results, f, indent=4)
        
        self.logger.info(f"Results saved to {final_output_path}")

# ================ MAIN EXECUTION ================
if __name__ == "__main__":
    
    class Args:
        dataset_path = "dataset"
        output_dir = "./classification_study_results"
        epochs = 5
        lr = 1e-6
        batch_size = 4
        grad_accum_steps = 4          
        max_length = DEFAULT_MAX_LENGTH
        seed = 42
        use_lora = True
        use_quantization = True

    args = Args()
    
    main_output_dir = os.path.join(args.output_dir, f"input_length_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(main_output_dir, exist_ok=True)
    
    all_experiment_results = {}
    
    print(f"\n{'='*80}")
    print(f"STARTING INPUT LENGTH ABLATION STUDY")
    print(f"Testing lengths: {INPUT_LENGTH_EXPERIMENTS}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"LoRA: {args.use_lora}, 4-bit Quantization: {args.use_quantization}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"{'='*80}")

    for model_key, model_config in MODEL_CONFIGS.items():
        print(f"{'='*60}") 
        print(f"TRAINING MODEL: {model_key} - {model_config['name']}")
        print(f"{'='*60}") 
        
        model_results = {}
        
        for max_length in INPUT_LENGTH_EXPERIMENTS:
            print(f"\n{'='*50}")
            print(f"EXPERIMENT: {model_key} - MAX_LENGTH = {max_length}")
            print(f"{'='*50}")
            
            args.max_length = max_length
            try:
                trainer = ResearchTrainer(
                    args, 
                    max_length, 
                    model_config, 
                    model_key,
                    use_lora=args.use_lora,
                    use_quantization=args.use_quantization
                )
                
                experiment_results = trainer.train()
                model_results[max_length] = experiment_results
                
                test_metrics = experiment_results.get('test_metrics', {})
                print(f"\n📊 RESULTS for {model_key} max_length={max_length}:")
                print(f"  ✅ Accuracy: {test_metrics.get('eval_accuracy', 'N/A'):.4f}")
                print(f"  ✅ Macro F1: {test_metrics.get('eval_macro_f1', 'N/A'):.4f}")
                if torch.cuda.is_available():
                    memory_used = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  💾 Memory used: {memory_used:.2f} GB")
                    
                    # Reset memory stats for next experiment
                    torch.cuda.reset_peak_memory_stats()
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "cuda out of memory" in error_msg:
                    print(f"🚨 OOM ERROR at {max_length} tokens for {model_key}! Skipping remaining lengths.")
                    model_results[max_length] = {"error": "GPU Out of Memory"}
                    oom_encountered = True
                else:
                    print(f"❌ Runtime error in {model_key} max_length={max_length}: {e}")
                    model_results[max_length] = {"error": f"RuntimeError: {str(e)}"}
            except Exception as e:
                print(f"❌ Unexpected error in {model_key} max_length={max_length}: {e}")
                model_results[max_length] = {"error": f"Unexpected: {str(e)}"}
            finally:
                # Safe cleanup
                try:
                    if 'trainer' in locals():
                        del trainer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception as e:
                    print(f"⚠️  Cleanup warning: {e}")
        
        all_experiment_results[model_key] = model_results

    # Comparative analysis
    print(f"\n{'='*80}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    comparative_results = {}
    for model_key, model_results in all_experiment_results.items():
        comparative_results[model_key] = {}
        
        for max_length, results in model_results.items():
            test_metrics = results.get('test_metrics', {})
            training_stats = results.get('training_stats', {})
            error_info = results.get('error', None)
            
            comparative_results[model_key][max_length] = {
                'test_metrics': {
                    'accuracy': test_metrics.get('eval_accuracy', None),
                    'macro_f1': test_metrics.get('eval_macro_f1', None),
                },
                'training_stats': {
                    'train_runtime': training_stats.get('train_runtime', None),
                    'train_samples_per_second': training_stats.get('train_samples_per_second', None),
                },
                'optimizations': {
                    'lora': args.use_lora,
                    'quantization': args.use_quantization
                },
                'error': error_info
            }
            
            print(f"\n{model_key} - Max Length {max_length}:")
            if error_info:
                print(f"  ❌ Error: {error_info}")
            else:
                print(f"  ✅ Accuracy: {test_metrics.get('eval_accuracy', 'N/A'):.4f}")
                print(f"  ✅ Macro F1: {test_metrics.get('eval_macro_f1', 'N/A'):.4f}")
                if 'train_runtime' in training_stats:
                    print(f"  ⏱️  Training Time: {training_stats.get('train_runtime', 'N/A'):.2f}s")

    comparative_path = os.path.join(main_output_dir, "comparative_analysis.json")
    with open(comparative_path, 'w') as f:
        json.dump(comparative_results, f, indent=4)
        
    print(f"\n📁 Comparative analysis saved to: {comparative_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"STUDY COMPLETED SUMMARY")
    print(f"{'='*80}")
    successful_experiments = 0
    total_experiments = 0
    
    for model_key, model_results in all_experiment_results.items():
        for max_length, results in model_results.items():
            total_experiments += 1
            if 'error' not in results or results['error'] is None:
                successful_experiments += 1
                
    print(f"✅ Successful experiments: {successful_experiments}/{total_experiments}")
    print(f"📊 Output directory: {main_output_dir}")

    print(f"{'='*80}\n")
