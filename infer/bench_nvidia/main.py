# ==========================================
# Nvidia_GPU / CPUの推論性能を測定するプログラム
# ==========================================

import time
import platform
import torch
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 設定エリア
# ==========================================

MODELS = [
    "unsloth/gpt-oss-20b",
]

LOOP_COUNT = 3

USE_4BIT = True  # ★ 4bit量子化ON/OFF切替

MAX_SEQ_LENGTH = 32768
MAX_NEW_TOKENS = 512

PROMPTS = [
    "Explain the history of the Internet in detail, covering its origins, key milestones, and future trends.",
    "Write a creative fantasy story about a clockmaker who controls time, at least 500 words.",
    "Explain the theory of relativity to a high school student using simple analogies and detailed examples.",
    "Generate a comprehensive Python code example for a simple Snake game using Pygame library, including comments.",
    "Describe the step-by-step process of photosynthesis and its importance to the global ecosystem."
]

# ==========================================
# デバイス判定
# ==========================================

def get_device_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Detected: {name}")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU Detected (MPS)")
        return "mps"
    else:
        print("⚠️ GPU NOT Detected. Using CPU.")
        return "cpu"

# ==========================================
# ウォームアップ
# ==========================================

def perform_warmup(model, tokenizer, device):
    print(f"\n{'='*20} Warming up... {'='*20}")
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=tokenizer.eos_token_id
        )

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("Warmup completed.\n")

# ==========================================
# 計測本体
# ==========================================

def measure_performance(model_id):
    device = get_device_info()

    print(f"\n{'='*60}")
    print(f"Loading Model: {model_id}")
    print(f"4bit Quantization: {'ON' if USE_4BIT else 'OFF'}")
    print(f"{'='*60}")

    try:
        # ------------------------------
        # 4bit量子化ON（CUDA限定）
        # ------------------------------
        if USE_4BIT and device == "cuda":
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=True,
            )
            model.eval()

        # ------------------------------
        # 通常ロード
        # ------------------------------
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            if device == "cpu":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    device_map="cpu"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype="auto",
                    device_map="auto"
                )

            model.eval()

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return

    perform_warmup(model, tokenizer, device)

    for loop_idx in range(LOOP_COUNT):
        print(f"--- Loop {loop_idx + 1}/{LOOP_COUNT} ---")

        loop_total_tokens = 0
        loop_total_time = 0.0

        for i, prompt in enumerate(PROMPTS):
            print(f"\n[Prompt {i+1}]")

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id
                )

            end = time.perf_counter()

            total_tokens = outputs.shape[1]
            duration = end - start
            tps = total_tokens / duration if duration > 0 else 0

            loop_total_tokens += total_tokens
            loop_total_time += duration

            text_snippet = tokenizer.decode(outputs[0], skip_special_tokens=True)[:100]

            print(f"Generated (Snippet): {text_snippet}...")
            print("-"*40)
            print(f"Total Tokens : {total_tokens}")
            print(f"Duration     : {duration:.2f} sec")
            print(f"TPS          : {tps:.2f} tokens/sec")

        # ==========================================
        # ★ 全体TPS表示
        # ==========================================
        overall_tps = loop_total_tokens / loop_total_time if loop_total_time > 0 else 0

        print(f"\n{'='*50}")
        print(f"Loop {loop_idx + 1} Summary")
        print(f"Total Tokens : {loop_total_tokens}")
        print(f"Total Time   : {loop_total_time:.2f} sec")
        print(f"Overall TPS  : {overall_tps:.2f} tokens/sec")
        print(f"{'='*50}")

# ==========================================
# エントリポイント
# ==========================================

if __name__ == "__main__":
    for model in MODELS:
        measure_performance(model)
