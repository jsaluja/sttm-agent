import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

REQUIREMENTS = [
    "accelerate",
    "torch==2.6.0",
    "datasets[audio]",
    "huggingface_hub[hf_xet]",
    "jiwer",
    "librosa",
    "transformers",
    "wandb",
    "soundfile"
]

for package in REQUIREMENTS:
    install(package)
    print(f"✅ Installed {package}")


# %%
import jiwer
import json
import math
import numpy as np
import pandas as pd
import pytz
import random
import re
import torch
import traceback
import wandb

from collections import defaultdict
from dataclasses import dataclass
from datasets import Audio, concatenate_datasets, load_dataset
from datetime import datetime
from huggingface_hub import hf_hub_download, HfApi, login, snapshot_download, upload_folder
from safetensors.torch import save_file as safe_save_file
from time import time
from transformers import TrainingArguments, Trainer, TrainerCallback, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import WAV2VEC2_ADAPTER_SAFE_FILE
from typing import Dict, List, Union

# %%
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_EVAL = 8
LEARNING_RATE = 1e-3
SEED = 42
USE_SMALL_DATASET = True  # Set to False for full dataset training
NUM_TRAIN_EPOCHS = 3 if USE_SMALL_DATASET else 12

# %%
BASE_MODEL = "facebook/mms-1b-all"

# %%
TARGET_LANGUAGE = "pan"

# %%
TRAIN_DATASET = 'jssaluja/rajinder_singh'
TEST_DATASETS = [
    'jssaluja/paath_anand_sahib',
    'jssaluja/paath_anand_sahib_2',
    'jssaluja/paath_barah_mah_tukhari',
    'jssaluja/paath_bavan_akhree',
    'jssaluja/paath_dakhanee_oankaar',
    'jssaluja/paath_jaitsree_ki_vaar',
    'jssaluja/paath_japji_sahib',
    'jssaluja/paath_salok_mehla_9',
    'jssaluja/paath_siddh_gosst',
    'jssaluja/paath_sukhmani_sahib',
]

# %%
def get_secret(secret_label):
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        secret_value = user_secrets.get_secret(secret_label)
    elif "COLAB_GPU" in os.environ:
        from google.colab import userdata
        secret_value = userdata.get(secret_label)
    else:
        secret_value = os.getenv(secret_label)

    return secret_value

# %%
HF_TOKEN = get_secret("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Error: Please set 'HF_TOKEN'")

# %%
login(HF_TOKEN)

# %%
# Get HuggingFace username from token
def get_hf_username(token):
    """Get HuggingFace username from token"""
    try:
        api = HfApi()
        user_info = api.whoami(token=token)
        username = user_info['name']
        print(f"🔑 HuggingFace username: {username}")
        return username
    except Exception as e:
        print(f"⚠️ Could not fetch HF username: {e}")
        print("⚠️ Falling back to 'jssaluja' as default")
        return "jssaluja"  # Fallback to your known username

HF_USERNAME = get_hf_username(HF_TOKEN)

# %%
# Constants and utilities
WANDB_PROJECT = "facebook-mms-1b-train"
METRICS = ['wer', 'wil', 'mer']

def get_wandb_run(repo_name, is_resuming_training=False):
    """Initialize WandB with consistent naming - same name for run_id and run_name"""
    WANDB_API_KEY = get_secret('WANDB_API_KEY')

    if WANDB_API_KEY:
        os.environ["WANDB_LOG_MODEL"] = "end"
        
        # Use the repo name directly as both run_id and run_name for consistency
        run_id = repo_name
        run_name = repo_name

        try:
            wandb.login(
                key=WANDB_API_KEY
            )
            
            if is_resuming_training:
                resume_mode = "allow"  # Allow resume if exists, create new if not
                print(f"   🔄 Resuming WandB training...")
            else:
                resume_mode = "never"  # Force new run
                print(f"   🆕 Starting new WandB training...")
            
            # Add timeout settings to prevent hanging
            wandb_settings = wandb.Settings(
                init_timeout=120,  # 2 minute timeout
                _disable_stats=True,  # Reduce overhead
                _disable_meta=True
            )
            
            wandb_run = wandb.init(
                id=run_id,
                name=run_name,
                job_type="debug" if USE_SMALL_DATASET else "train",
                project=WANDB_PROJECT,
                save_code=True,
                sync_tensorboard=False,
                tags=["asr", "resumed" if is_resuming_training else "fresh"],
                resume=resume_mode,
                settings=wandb_settings
            )

            print(f"   🔗 WandB Project: https://wandb.ai/{wandb_run.entity}/{WANDB_PROJECT}")
            print(f"   🚀 WandB Run: {wandb_run.url}")

            return wandb_run

        except Exception as wandb_init_exception:
            print('Error occured initializing wandb run ', wandb_init_exception)
            traceback.print_exc()
            # Return None to continue without WandB if initialization fails
            print("⚠️ Continuing training without WandB logging...")
            return None
    else:
        print("Error: Please set 'WANDB_API_KEY'")
        return None

# %%
def validate_training_config(repo_id, training_args_file, current_config, token):
    """Validate that checkpoint configuration matches current training settings and training is incomplete"""
    try:        
        # Download training_args.bin
        local_file = hf_hub_download(
            repo_id=repo_id,
            filename=training_args_file,
            token=token,
            repo_type="model"
        )
        
        # Load training arguments with PyTorch 2.6 compatibility
        try:
            # First try with weights_only=True (PyTorch 2.6 default)
            saved_args = torch.load(local_file, map_location='cpu', weights_only=True)
        except Exception as weights_only_error:
            try:
                # If that fails, try with weights_only=False for backward compatibility
                print(f"   ⚠️ weights_only=True failed, trying weights_only=False for compatibility")
                saved_args = torch.load(local_file, map_location='cpu', weights_only=False)
            except Exception as fallback_error:
                print(f"   ❌ Both loading methods failed:")
                print(f"      weights_only=True: {weights_only_error}")
                print(f"      weights_only=False: {fallback_error}")
                return False
        
        # Extract key parameters for comparison
        saved_config = {
            'num_train_epochs': getattr(saved_args, 'num_train_epochs', None),
            'per_device_train_batch_size': getattr(saved_args, 'per_device_train_batch_size', None),
            'learning_rate': getattr(saved_args, 'learning_rate', None),
            'output_dir': getattr(saved_args, 'output_dir', ''),
        }
        
        # Compare critical parameters
        config_matches = (
            saved_config['num_train_epochs'] == current_config['num_epochs'] and
            saved_config['per_device_train_batch_size'] == current_config['batch_size_train'] and
            abs(saved_config['learning_rate'] - current_config['learning_rate']) < 1e-6
        )
        
        print(f"   📋 Config comparison:")
        print(f"      Epochs: {saved_config['num_train_epochs']} vs {current_config['num_epochs']} ({'✅' if saved_config['num_train_epochs'] == current_config['num_epochs'] else '❌'})")
        print(f"      Batch size: {saved_config['per_device_train_batch_size']} vs {current_config['batch_size_train']} ({'✅' if saved_config['per_device_train_batch_size'] == current_config['batch_size_train'] else '❌'})")
        print(f"      Learning rate: {saved_config['learning_rate']} vs {current_config['learning_rate']} ({'✅' if abs(saved_config['learning_rate'] - current_config['learning_rate']) < 1e-6 else '❌'})")
        
        if not config_matches:
            return False
            
        # Check if training is already complete by examining trainer_state.json
        try:
            api = HfApi()
            files = api.list_repo_files(repo_id, token=token, repo_type="model")
            
            # Look for trainer_state.json in checkpoint-X folders
            import re
            checkpoint_pattern = re.compile(r'^checkpoint-(\d+)/trainer_state\.json$')
            checkpoint_nums = []
            
            for file in files:
                match = checkpoint_pattern.match(file)
                if match:
                    checkpoint_num = int(match.group(1))
                    checkpoint_nums.append((checkpoint_num, file))
            
            trainer_state_file = None
            if checkpoint_nums:
                # Sort by checkpoint number and get the latest one
                checkpoint_nums.sort(reverse=True)
                latest_checkpoint_num, trainer_state_file = checkpoint_nums[0]
                print(f"   📋 Found trainer_state.json in checkpoint-{latest_checkpoint_num}")
            
            if trainer_state_file:
                # Download and check trainer state
                trainer_state_local = hf_hub_download(
                    repo_id=repo_id,
                    filename=trainer_state_file,
                    token=token,
                    repo_type="model"
                )
                
                with open(trainer_state_local, 'r') as f:
                    trainer_state = json.load(f)
                
                current_epoch = trainer_state.get('epoch', 0)
                target_epochs = saved_config['num_train_epochs']
                
                print(f"   📊 Training progress:")
                print(f"      Current epoch: {current_epoch}")
                print(f"      Target epochs: {target_epochs}")
                
                # Check if training is complete
                if current_epoch >= target_epochs:
                    print(f"   ✅ Training complete ({current_epoch}/{target_epochs})")
                    return False  # Don't resume completed training
                else:
                    remaining_epochs = target_epochs - current_epoch
                    print(f"   🔄 Training incomplete ({remaining_epochs} epochs remaining)")
                    return True
            else:
                print(f"   ⚠️ No trainer_state.json found, assuming incomplete")
                return True
                
        except Exception as state_error:
            print(f"   ⚠️ Could not check training progress: {state_error}")
            print(f"   ⚠️ Assuming training is incomplete")
            return True
        
    except Exception as e:
        print(f"   ❌ Error validating config: {e}")
        return False

# %%
# PHASE 1: SETUP - Basic configuration and authentication
pacific_tz = pytz.timezone('America/Los_Angeles')
now = datetime.now(pacific_tz)
session_timestamp = now.strftime("%Y%m%d_%H%M%S")  # Compact format: 20250804_143022 (UTC)

print(f"🏷️  Session Timestamp: {session_timestamp}")

# Create current configuration for repo matching
current_config = {
    'train_dataset': TRAIN_DATASET,
    'num_epochs': NUM_TRAIN_EPOCHS,
    'num_test_datasets': len(TEST_DATASETS),
    'batch_size_train': BATCH_SIZE_TRAIN,
    'learning_rate': LEARNING_RATE,
    'use_small_dataset': USE_SMALL_DATASET
}

# %%
# PHASE 2: FINALIZE EPOCH/RESUME vs FRESH - Determine training mode early
print(f"🔍 Determining training mode (resume vs fresh)...")

def generate_repo_base_pattern(train_dataset, num_epochs, num_test_datasets):
    """Generate the base repository pattern (without timestamp) for consistent naming"""
    return f"fb-mms-1b-cleaned-{train_dataset.replace('/','_')}-epochs-{num_epochs}-test-datasets-{num_test_datasets}"

def find_resumable_checkpoint_early(hf_username, token, current_config):
    """Complete check to find the best incomplete checkpoint to resume"""
    try:
        api = HfApi()
        
        # Create base pattern without timestamp
        base_pattern = generate_repo_base_pattern(
            current_config['train_dataset'], 
            current_config['num_epochs'], 
            current_config['num_test_datasets']
        )
        
        print(f"   🔍 Searching for resumable repos matching pattern: {base_pattern}-*")
        
        # List all user's repos
        repos = api.list_models(author=hf_username, token=token)
        matching_repos = []
        
        for repo in repos:
            repo_name = repo.modelId.split('/')[-1]
            if repo_name.startswith(base_pattern):
                suffix = repo_name[len(base_pattern):]
                if suffix.startswith('-') and len(suffix) >= 16:
                    timestamp_part = suffix[1:16]
                    if (len(timestamp_part) == 15 and 
                        timestamp_part[:8].isdigit() and 
                        timestamp_part[8] == '_' and 
                        timestamp_part[9:15].isdigit()):
                        
                        remaining_suffix = suffix[16:]
                        if current_config['use_small_dataset']:
                            if remaining_suffix == '-small':
                                matching_repos.append(repo.modelId)
                        else:
                            if remaining_suffix == '':
                                matching_repos.append(repo.modelId)
        
        print(f"   🔍 Found {len(matching_repos)} potential resumable repos")
        
        if not matching_repos:
            print(f"   🆕 Fresh mode: No matching repos found")
            return False, None, None
            
        # Sort repos by timestamp (newest first) to prioritize recent runs
        matching_repos.sort(reverse=True)
        
        # Print all found repos for transparency
        for i, repo_id in enumerate(matching_repos):
            repo_name = repo_id.split('/')[-1]
            # Look for timestamp pattern: YYYYMMDD_HHMMSS (15 characters)
            import re
            timestamp_match = re.search(r'(\d{8}_\d{6})', repo_name)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                print(f"   📋 Found repo {i+1}: https://huggingface.co/{repo_id} ({timestamp})")
            else:
                print(f"   📋 Found repo {i+1}: https://huggingface.co/{repo_id}")
        
        print(f"   🔍 Validating repos for incomplete checkpoints...")
        
        # Helper function for detailed validation
        def validate_resume_checkpoint(repo_id, current_config, token):
            """Detailed validation of resume checkpoint"""
            try:
                api = HfApi()
                files = api.list_repo_files(repo_id, token=token, repo_type="model")
                
                # Look for checkpoint-X folders only
                import re
                checkpoint_pattern = re.compile(r'^checkpoint-(\d+)/training_args\.bin$')
                checkpoint_nums = []
                
                for file in files:
                    match = checkpoint_pattern.match(file)
                    if match:
                        checkpoint_num = int(match.group(1))
                        checkpoint_nums.append((checkpoint_num, f'checkpoint-{checkpoint_num}'))
                
                if not checkpoint_nums:
                    print(f"   ❌ No checkpoint-X/training_args.bin found")
                    return False
                
                # Sort by checkpoint number and get the latest one
                checkpoint_nums.sort(reverse=True)
                latest_checkpoint_num, checkpoint_dir = checkpoint_nums[0]
                training_args_file = f'{checkpoint_dir}/training_args.bin'
                print(f"   📋 Found latest checkpoint: {checkpoint_dir}")
                
                # Validate configuration compatibility and training completion
                if validate_training_config(repo_id, training_args_file, current_config, token):
                    # Check checkpoint completeness
                    required_files = [
                        f"{checkpoint_dir}/optimizer.pt",
                        f"{checkpoint_dir}/scheduler.pt", 
                        f"{checkpoint_dir}/trainer_state.json"
                    ]
                    
                    # Check for model files
                    model_files = [f for f in files if f.startswith(f'{checkpoint_dir}/') and ('pytorch_model.bin' in f or 'model.safetensors' in f)]
                    if not model_files:
                        print(f"   ❌ No model file found")
                        return False
                    
                    missing_files = [f for f in required_files if f not in files]
                    if missing_files:
                        print(f"   ❌ Incomplete checkpoint: {len(missing_files)} files missing")
                        return False
                    
                    print(f"   ✅ Complete checkpoint: {checkpoint_dir}")
                    return checkpoint_dir  # Return the checkpoint directory name
                else:
                    print(f"   ❌ Incompatible or complete")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Error validating checkpoint: {e}")
                return False
        
        # Try to find an incomplete repo
        for repo_id in matching_repos:
            repo_name = repo_id.split('/')[-1]
            print(f"   🔍 Validating: https://huggingface.co/{repo_id}")
            
            checkpoint_result = validate_resume_checkpoint(repo_id, current_config, token)
            if checkpoint_result:
                # Found a good incomplete checkpoint
                checkpoint_dir = checkpoint_result if isinstance(checkpoint_result, str) else "last-checkpoint"
                print(f"   ✅ Found incomplete checkpoint: {checkpoint_dir}")
                print(f"   🔗 Repository: https://huggingface.co/{repo_id}")
                return True, repo_name, repo_id, checkpoint_dir
        
        print(f"   ❌ All repos are either complete or incompatible")
        return False, None, None, None
            
    except Exception as e:
        print(f"   ❌ Error in repo check: {e}")
        return False, None, None, None

# Determine training mode early with complete validation
result = find_resumable_checkpoint_early(HF_USERNAME, HF_TOKEN, current_config)
if len(result) == 4:
    is_resuming, potential_repo_name, potential_model_id, resumable_checkpoint_dir = result
elif len(result) == 3:
    is_resuming, potential_repo_name, potential_model_id = result
    resumable_checkpoint_dir = None
else:
    is_resuming, potential_repo_name, potential_model_id, resumable_checkpoint_dir = False, None, None, None

# %%
# PHASE 3: IDENTIFY REPO/RUN IDs - Finalize naming before dataset loading
if is_resuming and potential_repo_name:
    REPO_NAME = potential_repo_name
    FINE_TUNED_MODEL_ID = potential_model_id
    # resumable_checkpoint_dir is now set from the function
    print(f"🔄 RESUME MODE: {resumable_checkpoint_dir}")
    print(f"   🔗 Repository: https://huggingface.co/{FINE_TUNED_MODEL_ID}")
else:
    # Create new timestamped repo name
    BASE_REPO_NAME = f"{generate_repo_base_pattern(TRAIN_DATASET, NUM_TRAIN_EPOCHS, len(TEST_DATASETS))}-{session_timestamp}"
    if USE_SMALL_DATASET:
        BASE_REPO_NAME = f"{BASE_REPO_NAME}-small"
    REPO_NAME = BASE_REPO_NAME
    FINE_TUNED_MODEL_ID = f"{HF_USERNAME}/{REPO_NAME}"
    resumable_checkpoint_dir = None
    is_resuming = False  # Ensure consistency
    print(f"🆕 FRESH MODE: New repository")
    print(f"   🔗 Repository: https://huggingface.co/{FINE_TUNED_MODEL_ID}")

print(f"📋 Training Configuration:")
print(f"   Mode: {'🔄 RESUME' if is_resuming else '🆕 FRESH'}")
print(f"   Checkpoint: {resumable_checkpoint_dir if resumable_checkpoint_dir else 'None'}")

# %%
# PHASE 4: LOAD DATASET - Download and prepare datasets
print(f"📥 PHASE 4: Loading datasets...")

os.environ["HF_DATASETS_CACHE"] = "/root/.cache/huggingface/datasets"

tic = time()
audio_dataset_train = load_dataset(TRAIN_DATASET, split="train", token=HF_TOKEN, cache_dir=os.environ["HF_DATASETS_CACHE"])
audio_dataset_test = concatenate_datasets(
    [
        load_dataset(test_dataset, split="train+validation+test", token=HF_TOKEN, cache_dir=os.environ["HF_DATASETS_CACHE"]) for test_dataset in TEST_DATASETS
    ]
)

audio_dataset_train = audio_dataset_train.cast_column("audio", Audio(sampling_rate=16000))
audio_dataset_test = audio_dataset_test.cast_column("audio", Audio(sampling_rate=16000))

toc = time()
dataset_load_time = toc - tic

print(f'   ✅ Downloaded dataset in {dataset_load_time:.2f} seconds')
print(f'   📊 {audio_dataset_train = }, {audio_dataset_test = }')

# Dataset reduction based on USE_SMALL_DATASET flag
if USE_SMALL_DATASET:
    print('   🔄 Selecting 1/100th of dataset for faster testing...')
    original_train_size = len(audio_dataset_train)
    original_test_size = len(audio_dataset_test)

    audio_dataset_train = audio_dataset_train.select(range(len(audio_dataset_train) // 500))
    audio_dataset_test = audio_dataset_test.select(range(len(audio_dataset_test) // 500))

    print(f'   ✅ Dataset reduced: train {original_train_size} -> {len(audio_dataset_train)}, test {original_test_size} -> {len(audio_dataset_test)}')
else:
    print('   📊 Using full dataset for training...')

# %%
# PHASE 5: PREPROCESS - Clean data, build vocabulary, and prepare for training
# print(f"🔄 PHASE 5: Preprocessing datasets...")

# Enhanced text processing utilities from utils.py
REPLACE_DIGITS_WITH_WORDS = {
    "ਮਹਲਾ ੩ ਮਹਲਾ ਤੀਜਾ": "ਮਹਲਾ ਤੀਜਾ",
    "ਮਹਲਾ ੩ ਤੀਜਾ": "ਮਹਲਾ ਤੀਜਾ",
    "੧ ਪਹਿਲਾ": "ਪਹਿਲਾ",
    "੩ ਤੀਜਾ": "ਤੀਜਾ",
    "੪ ਚਉਥਾ": "ਚਉਥਾ",
    "ਮਹਲਾ ੧": "ਮਹਲਾ ਪਹਿਲਾ",
    "ਮਹਲੁ ੧": "ਮਹਲੁ ਪਹਿਲਾ",
    "ਮਹਲ ੧": "ਮਹਲ ਪਹਿਲਾ",
    "ਮਰਦਾਨਾ ੧": "ਮਰਦਾਨਾ ਪਹਿਲਾ",
    "ਮਹਲਾ ਪਹਿਲਾ ੧": "ਮਹਲਾ ਪਹਿਲਾ",
    "ਮਹਲਾ ੨": "ਮਹਲਾ ਦੂਜਾ",
    "ਮਹਲਾ ੩": "ਮਹਲਾ ਤੀਜਾ",
    "ਸੋਲਹੇ ੩": "ਸੋਲਹੇ ਤੀਜਾ",
    "ਮਹਲਾ ੪": "ਮਹਲਾ ਚਉਥਾ",
    "ਮਹਲੇ ੪": "ਮਹਲੇ ਚਉਥੇ",
    "ਮਹਲਾ ੫": "ਮਹਲਾ ਪੰਜਵਾ",
    "ਮਹਲੁ ੫": "ਮਹਲੁ ਪੰਜਵਾ",
    "ਮਾਲਾ ੫": "ਮਾਲਾ ਪੰਜਵਾ",
    "ਪਉੜੀ ੫": "ਪਉੜੀ ਪੰਜਵਾ",
    "ਮਹਲਾ ੯": "ਮਹਲਾ ਨੌਵਾਂ",
    "ਮਃ ੧": "ਮਹਲਾ ਪਹਿਲਾ",
    "ਮਃ ੨": "ਮਹਲਾ ਦੂਜਾ",
    "ਮਃ ੩": "ਮਹਲਾ ਤੀਜਾ",
    "ਮਃ ੪": "ਮਹਲਾ ਚਉਥਾ",
    "ਮਃ ੫":"ਮਹਲਾ ਪੰਜਵਾ",
    "ਦੇਵਗੰਧਾਰੀ ੫": "ਦੇਵਗੰਧਾਰੀ ਪੰਜਵਾ",
    "ਘਰੁ ੧੦": "ਘਰੁ ਦਸਵਾਂ",
    "ਘਰੁ ੧੧": "ਘਰੁ ਗਿਆਰਵਾਂ",
    "ਘਰੁ ੧੨": "ਘਰੁ ਬਾਰਹਵਾਂ",
    "ਘਰੁ ੧੩": "ਘਰੁ ਤੇਰਵਾਂ",
    "ਘਰੁ ੧੪": "ਘਰੁ ਚੌਦਵਾਂ",
    "ਘਰੁ ੧੫": "ਘਰੁ ਪੰਦਰਵਾਂ",
    "ਘਰੁ ੧੬ ਕੇ ੨": "ਘਰੁ ਸੋਲ੍ਹਵੇਂ ਕੇ ਦੋ",
    "ਘਰੁ ੧੬": "ਘਰੁ ਸੋਲਵਾਂ",
    "ਘਰੁ ੧੭": "ਘਰੁ ਸਤਾਰਵਾਂ",
    "ਘਰੁ ੧੮": "ਘਰੁ ਅਠਾਰਵਾਂ",
    "ਘਰੁ ੧": "ਘਰੁ ਪਹਿਲਾ",
    "ਘਰੁ ੨": "ਘਰੁ ਦੂਜਾ",
    "ਘਰੁ ੩": "ਘਰੁ ਤੀਜਾ",
    "ਘਰੁ ੪": "ਘਰੁ ਚਉਥਾ",
    "ਘਰੁ ੫": "ਘਰੁ ਪੰਜਵਾ",
    "ਘਰੁ ੬ ਕੇ ੩": "ਘਰੁ ਛੇਵੇਂ ਕੇ ਤਿੰਨ",
    "ਘਰੁ ੬": "ਘਰੁ ਛੇਵਾਂ",
    "ਘਰੁ ੭": "ਘਰੁ ਸੱਤਵਾਂ",   
    "ਘਰੁ ੮": "ਘਰੁ ਅੱਠਵਾਂ",
    "ਘਰੁ ੯": "ਘਰੁ ਨੌਵਾਂ",
    "ਘਰੁ ਦੂਜਾ ੨": "ਘਰੁ ਦੂਜਾ",
    "ਤਿਪਦੇ ੨": "ਤਿਪਦੇ ਦੋ",
    "ਗਉੜੀ ੧੧": "ਗਉੜੀ ਗਿਆਰਵੀਂ",
    "ਗਉੜੀ ੧੨": "ਗਉੜੀ ਬਾਰਵੀਂ",
    "ਪੂਰਬੀ ੧੨": "ਪੂਰਬੀ ਬਾਰਵੀਂ",
    "ਗਉੜੀ ੧੩": "ਗਉੜੀ ਤੇਰਵੀਂ",
    "ਗਉੜੀ ੯": "ਗਉੜੀ ਨੌਵੀਂ",
    "ਚਉਪਦੇ ੧੪": "ਚਉਪਦੇ ਚੌਦਾਂ",
    "ਇਕਤੁਕੇ ੨": "ਇਕਤੁਕੇ ਦੋ",
    "ਇਕਤੁਕੇ ੪": "ਇਕਤੁਕੇ ਚਾਰ",
    "ਦੁਪਦਾ ੧": "ਦੁਪਦਾ ਇੱਕ",
    "ਛਕਾ ੧": "ਛਕਾ ਇੱਕ",
    "ਛਕੇ ੨": "ਛਕੇ ਦੋ",
    "ਛਕੇ ੩": "ਛਕੇ ਤਿੰਨ",
    "ਦੁਤੁਕੇ ੯": "ਦੁਤੁਕੇ ਨੌੰ",
    "ਕੇ ੭": "ਕੇ ਸੱਤ",
    "ਪਹਿਲੇ ਕੇ ੧": "ਪਹਿਲੇ ਕੇ",
    "ਦੂਜੇ ਕੇ ੨": "ਦੂਜੇ ਕੇ",
    "ਤੀਜੇ ਕੇ ੩": "ਤੀਜੇ ਕੇ",
    "ਚਉਥੇ ਕੇ ੪": "ਚਉਥੇ ਕੇ",
    "ਪੰਜਵੇ ਕੇ ੫": "ਪੰਜਵੇ ਕੇ",
    "ਇਕਤੁਕਾ ੧": "ਇਕਤੁਕਾ ਇੱਕ",
    "ਤਿਪਦੇ ੮": "ਤਿਪਦੇ ਅੱਠ",
    "ਦੁਤੁਕੇ ੭": "ਦੁਤੁਕੇ ਸੱਤ",
    "ਪੰਚਪਦੇ ੯": "ਪੰਚਪਦੇ ਨੌਂ",
    "ਦੁਤੁਕੇ ੫": "ਦੁਤੁਕੇ ਪੰਜ"
}


def remove_numbers_between_separators(text: str, replacement: str, pattern: str):
    """Remove numbers between separators (e.g., ॥੧॥) and normalize separators"""
    pattern = re.compile(pattern)
    matches = re.findall(pattern, text)

    while matches:
        text = re.sub(pattern, replacement, text)
        matches = re.findall(pattern, text)

    # English has ||
    if len(replacement) > 1:
        return text
    
    # Remove multiple consecutive separators
    return re.sub(f'{replacement}+', replacement, text)


def parse_punjabi(text: str) -> str:
    """
    Enhanced Punjabi text cleaning:
    - Removes Punjabi numbers enclosed between ॥ and ॥ from the given text
    - Expands abbreviated forms (e.g., ਮਃ ੧ -> ਮਹਲਾ ਪਹਿਲਾ)
    - Normalizes spacing and separators
    """
    # Remove Punjabi numbers between separators
    text = remove_numbers_between_separators(
        text,
        '॥',
        r"॥[੦੧੨੩੪੫੬੭੮੯]+॥"
    ).replace("॥", "").strip()

    # Expand abbreviated digit forms to full words
    for key, value in REPLACE_DIGITS_WITH_WORDS.items():
        text = text.replace(key, value)

    # Clean up any double spaces created by digit removal
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess(batch):
    """Enhanced dataset preprocessing - clean text data with comprehensive Punjabi processing"""    
    # Check which field contains the original text - different datasets have different field names
    if "orig_text" in batch:
        # Training dataset has orig_text field
        original_text = batch["orig_text"]
    else:
        # Test dataset only has sentence field
        original_text = batch["sentence"]
    
    # Apply preprocessing and update the sentence field
    batch["sentence"] = parse_punjabi(original_text)
    return batch


print('   🧹 Cleaning text data...')
tic = time()
audio_dataset_train = audio_dataset_train.map(preprocess)
audio_dataset_test = audio_dataset_test.map(preprocess)
toc = time()
dataset_preprocess_time = toc - tic

print(f'   ✅ Preprocessed dataset in {dataset_preprocess_time:.2f} seconds')

# Calculate train_steps early since we now have final dataset sizes
train_steps = NUM_TRAIN_EPOCHS * math.ceil((len(audio_dataset_train)) / BATCH_SIZE_TRAIN)
print(f'   📊 {train_steps = }')

# Show random dataset elements for verification
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    print(df)

print('   📋 Sample preprocessed data:')
show_random_elements(audio_dataset_train.remove_columns(["audio"]), num_examples=10)

# Build vocabulary from preprocessed dataset
print('   🔤 Building vocabulary...')
def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = audio_dataset_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=audio_dataset_train.column_names)

vocab_list = list(set(vocab_train["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict = dict(sorted(vocab_dict.items(), key=lambda item: item[1]))

print(f'   📊 Initial vocabulary: {len(vocab_dict)} characters')

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
vocab_dict = dict(sorted(vocab_dict.items(), key=lambda item: item[1]))

print(f'   ✅ Final vocabulary: {len(vocab_dict)} characters (including UNK and PAD)')

new_vocab_dict = {TARGET_LANGUAGE: vocab_dict}

vocab_object = json.dumps(new_vocab_dict, ensure_ascii=False, indent=4)
with open('vocab.json', 'w', encoding="utf-8") as vocab_file:
    vocab_file.write(vocab_object)

# Note: Checkpoint validation was already completed in Phase 2 if resuming

# %%
# PHASE 6: TRAIN - Setup model and begin training
print(f"🚀 PHASE 6: Setting up training...")

# Initialize WandB with final repo name (after all validations are complete)
print(f"🎯 Initializing WandB...")
wandb_run = get_wandb_run(REPO_NAME, is_resuming_training=is_resuming)

# Create tokenizer and processor from vocabulary
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", target_lang=TARGET_LANGUAGE)

# Push tokenizer to hub (using final repo name)
tokenizer.push_to_hub(REPO_NAME, token=HF_TOKEN, private=True)

# Create processor for audio feature extraction and tokenization
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Prepare final dataset with audio features and labels
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch

print("   🔄 Converting datasets to model format...")
tic = time()
audio_dataset_train = audio_dataset_train.map(prepare_dataset, remove_columns=audio_dataset_train.column_names)
audio_dataset_test = audio_dataset_test.map(prepare_dataset, remove_columns=audio_dataset_test.column_names)
toc = time()
dataset_prepare_time = toc - tic

print(f'   ✅ Prepared dataset in {dataset_prepare_time:.2f} seconds')

# %%
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

# %%
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


def detect_changes(alignment, label_str, pred_str):
    # Split the alignment into lines
    lines = alignment.split("\n")

    print('Prediction samples')
    for line in lines[:54]:
        print(line)

    # Initialize dictionaries to store the substitutions, insertions, and deletions
    substitutions = defaultdict(lambda: defaultdict(int))
    insertions = defaultdict(int)
    deletions = defaultdict(int)
    
    # Initialize lists to store sentence pairs and specific word errors with context
    substitution_sentences = []
    insertion_sentences = []
    deletion_sentences = []
    
    # Lists to store specific word errors with sentence context
    substitution_word_details = []
    insertion_word_details = []
    deletion_word_details = []

    # Iterate over the lines in the alignment
    sentence_idx = 0
    for i in range(len(lines)):
        # If the line starts with "REF:", it contains the reference sentence
        if lines[i].startswith("REF:"):
            ref_words = lines[i].split()[1:]  # Skip first element REF:

            # The next line contains the hypothesis sentence
            hyp_words = lines[i+1].split()[1:]  # Skip first element HYP:

            # The line after that contains the alignment
            align_symbols = lines[i+2].split() # Get the alignment symbols
            
            # Track if this sentence has each error type
            has_substitution = False
            has_insertion = False
            has_deletion = False
            
            # index of alignment symbols
            a = 0
            for j in range(len(ref_words)):
                if ref_words[j] == hyp_words[j]:
                    continue

                # Get the alignment symbol at this position
                align_symbol = align_symbols[a]
                a += 1

                ref_word = ref_words[j]
                hyp_word = hyp_words[j]

                # Check the type of change (substitution, insertion, or deletion)
                if align_symbol == "S":
                    substitutions[ref_word][hyp_word] += 1
                    has_substitution = True
                    # Store specific word substitution with sentence context
                    if sentence_idx < len(label_str) and sentence_idx < len(pred_str):
                        substitution_word_details.append({
                            'original_word': ref_word,
                            'predicted_word': hyp_word,
                            'original_sentence': label_str[sentence_idx],
                            'predicted_sentence': pred_str[sentence_idx]
                        })
                elif align_symbol == "I":
                    insertions[hyp_word] += 1
                    has_insertion = True
                    # Store specific word insertion with sentence context
                    if sentence_idx < len(label_str) and sentence_idx < len(pred_str):
                        insertion_word_details.append({
                            'inserted_word': hyp_word,
                            'original_sentence': label_str[sentence_idx],
                            'predicted_sentence': pred_str[sentence_idx]
                        })
                elif align_symbol == "D":
                    deletions[ref_word] += 1
                    has_deletion = True
                    # Store specific word deletion with sentence context
                    if sentence_idx < len(label_str) and sentence_idx < len(pred_str):
                        deletion_word_details.append({
                            'deleted_word': ref_word,
                            'original_sentence': label_str[sentence_idx],
                            'predicted_sentence': pred_str[sentence_idx]
                        })
            
            # Add sentence pairs to respective lists if they have errors
            if sentence_idx < len(label_str) and sentence_idx < len(pred_str):
                if has_substitution:
                    substitution_sentences.append({
                        'label': label_str[sentence_idx],
                        'prediction': pred_str[sentence_idx]
                    })
                if has_insertion:
                    insertion_sentences.append({
                        'label': label_str[sentence_idx],
                        'prediction': pred_str[sentence_idx]
                    })
                if has_deletion:
                    deletion_sentences.append({
                        'label': label_str[sentence_idx],
                        'prediction': pred_str[sentence_idx]
                    })
            
            sentence_idx += 1

    # Sort the dictionaries by total count in descending order
    substitutions = {
        k: dict(sorted(v.items(), key=lambda item: item[1], reverse=True))
        for k, v in sorted(
            substitutions.items(), key=lambda item: sum(item[1].values()), reverse=True
        )
    }
    insertions = dict(
        sorted(insertions.items(), key=lambda item: item[1], reverse=True)
    )
    deletions = dict(
        sorted(deletions.items(), key=lambda item: item[1], reverse=True)
    )

    changes = {
        'substitutions': substitutions,
        'insertions': insertions,
        'deletions': deletions,
        'substitution_sentences': substitution_sentences,
        'insertion_sentences': insertion_sentences,
        'deletion_sentences': deletion_sentences,
        'substitution_word_details': substitution_word_details,
        'insertion_word_details': insertion_word_details,
        'deletion_word_details': deletion_word_details
    }
    return changes


# %%
def compute_metrics(pred):
    tic = time()
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False, skip_special_tokens=True)
        
    word_alignment = jiwer.process_words(label_str, pred_str)

    computed_metrics = {metric: getattr(word_alignment, metric) for metric in METRICS if hasattr(word_alignment, metric)}
    
    cer = jiwer.cer(label_str, pred_str)
    computed_metrics['cer'] = cer

    print('Dataset Metrics')
    print(pd.DataFrame.from_dict(computed_metrics, orient='index', columns=['Metric']))    
    

    toc = time()
    print(f'   ✅ Computed metrics in {(toc-tic):.2f} seconds')

    if wandb_run:
        wandb_run.log(computed_metrics, commit=True)
        
    return computed_metrics


model = Wav2Vec2ForCTC.from_pretrained(
    BASE_MODEL,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
)

# %%
model.init_adapter_layers()

# Next, we freeze all weights, **but** the adapter layers.
model.freeze_base_model()

adapter_weights = model._get_adapters()
for param in adapter_weights.values():
    param.requires_grad = True

class EpochProgressCallback(TrainerCallback):
    """Custom callback to print epoch progress"""
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        # state.epoch represents completed epochs (0 for first epoch, 1 for second, etc.)
        if state.epoch == 0:
            epoch_starting = 1  # Very first epoch
        else:
            # state.epoch represents completed epochs, so next epoch is +1
            epoch_starting = int(state.epoch) + 1
        total_epochs = args.num_train_epochs
        print(f"🚀 Starting Epoch  {epoch_starting:2d}/{total_epochs:2d}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        # state.epoch represents the epoch number that just completed (1-based)
        epoch_completed = int(state.epoch) if state.epoch is not None else 1
        total_epochs = args.num_train_epochs
        print(f"✅ Completed Epoch {epoch_completed:2d}/{total_epochs:2d}")

class TensorBoardUploadCallback(TrainerCallback):
    """Custom callback to upload TensorBoard logs at each checkpoint save"""
    
    def __init__(self, repo_id, token, tensorboard_dir, total_steps, total_epochs):
        self.repo_id = repo_id
        self.token = token
        self.tensorboard_dir = tensorboard_dir
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        
    def on_save(self, args, state, control, **kwargs):
        """Upload TensorBoard logs whenever a checkpoint is saved (now at epoch boundaries)"""
        try:
            if os.path.exists(self.tensorboard_dir):
                current_step = state.global_step
                # state.epoch represents the epoch number that just completed (1-based)
                epoch_completed = int(state.epoch) if state.epoch is not None else 1
                                
                upload_folder(
                    folder_path=self.tensorboard_dir,
                    repo_id=self.repo_id,
                    path_in_repo="runs",  # Standard TensorBoard directory name
                    token=self.token,
                    repo_type="model",
                    commit_message=f"Update TensorBoard logs at epoch {epoch_completed}"
                )
                print(f"✅ TensorBoard logs uploaded at epoch {epoch_completed:2d}/{self.total_epochs:2d} (step {current_step:4d}/{self.total_steps:4d})")
            else:
                print(f"⚠️ TensorBoard directory not found: {self.tensorboard_dir}")
        except Exception as e:
            print(f"❌ Failed to upload TensorBoard logs: {e}")
            # Don't fail training if upload fails

training_args = TrainingArguments(
    seed=SEED,
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    per_device_eval_batch_size=BATCH_SIZE_EVAL,    
    eval_strategy="epoch",  
    save_strategy="epoch",  
    logging_strategy="epoch", 
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    push_to_hub=True,
    push_to_hub_token=HF_TOKEN,
    hub_private_repo=True,
    hub_model_id=REPO_NAME,
    gradient_checkpointing=True,
    group_by_length=True,
    fp16=True,
    output_dir=REPO_NAME,
    warmup_steps=0, # Placeholder - will be dynamically calculated based on dataset size and resume status
    save_total_limit=3, 
    hub_strategy="all_checkpoints",
    metric_for_best_model="wer",
    greater_is_better=False,    
    load_best_model_at_end=True,
    logging_dir=f"./tensorboard_logs/{REPO_NAME}",
    report_to=["tensorboard", "wandb"] if wandb_run else ["tensorboard"],
    run_name=f"{REPO_NAME}-continuous",
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=audio_dataset_train,
    eval_dataset=audio_dataset_test,
    tokenizer=processor.feature_extractor,
    callbacks=[
        EpochProgressCallback(),  # Add epoch progress callback
        TensorBoardUploadCallback(  # Add TensorBoard upload callback
            repo_id=FINE_TUNED_MODEL_ID,
            token=HF_TOKEN,
            tensorboard_dir=f"./tensorboard_logs/{REPO_NAME}",
            total_steps=train_steps,
            total_epochs=NUM_TRAIN_EPOCHS
        )
    ],
)

print('Training')

# Final training configuration
print(f"📋 Final Training Configuration:")
print(f"   Mode: {'🔄 RESUME' if is_resuming else '🆕 FRESH'}")
print(f"   Repo: {FINE_TUNED_MODEL_ID}")
print(f"   Checkpoint: {resumable_checkpoint_dir if resumable_checkpoint_dir else 'None'}")
print(f"   Train samples: {len(audio_dataset_train)}")
print(f"   Test samples: {len(audio_dataset_test)}")
print(f"   Total steps: {train_steps}")

# Set checkpoint directory for training
checkpoint_dir = resumable_checkpoint_dir if is_resuming and resumable_checkpoint_dir else None

# Update training args to include WandB in report_to
training_args.report_to = ["tensorboard", "wandb"] if wandb_run else ["tensorboard"]

if is_resuming:
    print("📋 Resuming training: Disabling warmup to prevent learning rate jumps")
    warmup_steps_adjusted = 0
else:
    print("🆕 Fresh training: Using dataset-size-dependent warmup")
    # Calculate warmup as 5% of first epoch, with reasonable bounds
    steps_per_epoch = math.ceil(len(audio_dataset_train) / BATCH_SIZE_TRAIN)
    warmup_steps_adjusted = max(10, min(200, int(0.05 * steps_per_epoch)))
    print(f"   Dataset size: {len(audio_dataset_train)} samples")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Warmup steps: {warmup_steps_adjusted} ({warmup_steps_adjusted/steps_per_epoch*100:.1f}% of first epoch)")

# Update training args with adjusted warmup
training_args.warmup_steps = warmup_steps_adjusted
print(f"🔧 Warmup steps set to: {warmup_steps_adjusted}")

tic = time()

if checkpoint_dir:
    print(f"🎯 Found complete checkpoint {checkpoint_dir} on Hub. Downloading and resuming training...")
    
    try:
        # Download only the checkpoint directory
        print(f"📥 Downloading checkpoint {checkpoint_dir}...")
        local_checkpoint_dir = snapshot_download(
            repo_id=FINE_TUNED_MODEL_ID,
            token=HF_TOKEN,
            allow_patterns=f"{checkpoint_dir}/**",
            local_dir_use_symlinks=False
        )
        
        checkpoint_path = os.path.join(local_checkpoint_dir, checkpoint_dir)
        print(f"✅ Downloaded checkpoint to: {checkpoint_path}")
        
        # Verify checkpoint files exist locally
        required_local_files = [
            os.path.join(checkpoint_path, "optimizer.pt"),
            os.path.join(checkpoint_path, "scheduler.pt"),
            os.path.join(checkpoint_path, "trainer_state.json")
        ]
        
        for file_path in required_local_files:
            if os.path.exists(file_path):
                print(f"✅ Found: {os.path.basename(file_path)}")
            else:
                print(f"❌ Missing: {os.path.basename(file_path)}")
        
        # Read trainer state to get progress information
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            try:
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
                
                current_step = trainer_state.get('global_step', 0)
                completed_epochs = int(trainer_state.get('epoch', 0))
                next_epoch_to_start = completed_epochs + 1
                
                print(f"🚀 Resuming from checkpoint: step {current_step:4d}/{train_steps:4d}, epoch {completed_epochs:2d}/{NUM_TRAIN_EPOCHS:2d} → {next_epoch_to_start:2d}")
            except Exception as e:
                print(f"🚀 Resuming training from checkpoint: {checkpoint_path}")
                print(f"   (Could not read progress info: {e})")
        else:
            print(f"🚀 Resuming training from checkpoint: {checkpoint_path}")
        
        trainer.train(resume_from_checkpoint=checkpoint_path)
        
    except Exception as e:
        print(f"❌ Error downloading checkpoint: {e}")
        print("🔄 Falling back to training from scratch...")
        trainer.train()
        
else:
    if is_resuming:
        print(f"❌ Resume mode activated but no checkpoint found in: {FINE_TUNED_MODEL_ID}")
        print("🔄 This shouldn't happen - falling back to fresh training...")
    else:
        print("🆕 No complete checkpoint found on Hub, starting training from scratch...")
    trainer.train()

toc = time()

print(f"Model training time : {toc-tic}")


adapter_file = WAV2VEC2_ADAPTER_SAFE_FILE.format(TARGET_LANGUAGE)
adapter_file = os.path.join(training_args.output_dir, adapter_file)

safe_save_file(model._get_adapters(), adapter_file, metadata={"format": "pt"})

# %%
kwargs = {
    "dataset": TRAIN_DATASET,
    "finetuned_from": BASE_MODEL,
    "language": TARGET_LANGUAGE,
    "model_name": FINE_TUNED_MODEL_ID,
    "tags": ["hf-asr-leaderboard"],
    "tasks": "automatic-speech-recognition",
}
trainer.evaluate(audio_dataset_test)
trainer.push_to_hub(FINE_TUNED_MODEL_ID, **kwargs)

# Display final summary
print(f"🎉 Training completed successfully!")
print(f"   Hugging Face Repository: https://huggingface.co/{FINE_TUNED_MODEL_ID}")
print(f"   WandB Run: https://wandb.ai/{wandb_run.entity}/{WANDB_PROJECT}/runs/{REPO_NAME}" if wandb_run else f"   WandB Run: {REPO_NAME}")
print(f"   Session: {session_timestamp}")

if wandb_run is not None:
    wandb_run.finish()
