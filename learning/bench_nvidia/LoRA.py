# ==========================================
# Nvidia_GPU (Unsloth) 学習性能を測定するプログラム
# ==========================================

import time
import torch
import psutil
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 設定エリア
# ==========================================

# テスト対象モデル
MODEL_ID = "unsloth/gpt-oss-20b" # 学習テスト用に一般的なモデルを指定（変更可能）
DATA_FILE = "alpaca_cleaned_ja.json"

# 学習パラメータ設定
MAX_SEQ_LENGTH = 8192
USE_4BIT = True
LORA_RANK = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# 測定用設定
MAX_STEPS = 60         # 測定する学習ステップ数（エポックではなくステップ数で固定して時間を測る）
BATCH_SIZE = 12        # バッチサイズ
GRAD_ACCUMULATION = 5  # 勾配蓄積数

# ==========================================
# デバイス判定 & 初期チェック
# ==========================================

def check_environment():
    print(f"\n{'='*20} Environment Check {'='*20}")
    
    # Unslothによる学習は基本的にNVIDIA GPU（CUDA）が必須です
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Detected: {gpu_name} ({vram_total:.1f} GB VRAM)")
        return True
    else:
        print("❌ Error: Nvidia GPU not detected. Unsloth training requires CUDA.")
        return False

# ==========================================
# データセット準備
# ==========================================

def load_and_format_data(tokenizer):
    print("Loading dataset...")
    try:
        dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    except Exception as e:
        print(f"❌ Failed to load {DATA_FILE}: {e}")
        print("Please ensure 'test_data.json' exists in the current directory.")
        return None

    # プロンプトフォーマット関数 (Alpaca形式を想定)
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    EOS_TOKEN = tokenizer.eos_token 

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"Dataset loaded. Rows: {len(dataset)}")
    return dataset

# ==========================================
# 学習計測本体
# ==========================================

def measure_training_performance():
    if not check_environment():
        return

    print(f"\n{'='*60}")
    print(f"Loading Model for Training: {MODEL_ID}")
    print(f"4bit Quantization: {'ON' if USE_4BIT else 'OFF'}")
    print(f"{'='*60}")

    try:
        # 1. モデルとトークナイザーのロード
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_ID,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None,
            load_in_4bit = USE_4BIT,
        )

        # 2. LoRAアダプターの適用
        model = FastLanguageModel.get_peft_model(
            model,
            r = LORA_RANK,
            target_modules = TARGET_MODULES,
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth", # メモリ節約設定
            random_state = 3407,
            use_rslora = False,
            loftq_config = None,
        )
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return

    # 3. データセットの準備
    dataset = load_and_format_data(tokenizer)
    if dataset is None:
        return

    # 4. トレーナーの設定
    print(f"\n{'='*20} Starting Training Measurement {'='*20}")
    print(f"Target Steps: {MAX_STEPS}")
    
    training_args = TrainingArguments(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUMULATION,
        warmup_steps = 5,
        max_steps = MAX_STEPS, # エポックではなくステップ数で測定
        learning_rate = 4e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        output_dir = "outputs_perf_test",
        optim = "adamw_8bit",
        report_to = "none", # WandBなどを無効化
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = training_args,
    )

    # 5. メモリ統計のリセット
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 6. 学習実行と時間計測
    start_time = time.perf_counter()
    
    trainer_stats = trainer.train()
    
    end_time = time.perf_counter()

    # ==========================================
    # 結果表示
    # ==========================================
    
    duration = end_time - start_time
    total_steps = trainer_stats.global_step
    
    print(f"\n{'='*60}")
    print("Training Performance Results")
    print(f"{'='*60}")
    
    print(f"Model         : {MODEL_ID}")
    print(f"Training Time : {duration:.2f} seconds")
    print(f"Total Steps   : {total_steps}")
    print(f"Steps/Sec     : {total_steps / duration:.4f} steps/sec")
    print(f"Sec/Step      : {duration / total_steps:.4f} seconds/step")
    print(f"{'='*60}")

# ==========================================
# エントリポイント
# ==========================================

if __name__ == "__main__":
    measure_training_performance()