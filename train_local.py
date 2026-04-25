#!/usr/bin/env python3
"""
ClusterOps Local Training Simulation.

Runs a small LLM (Qwen 0.5B) against the ClusterOps environment locally
to verify the training loop works end-to-end on CPU or GPU.

Configuration is done via environment variables:
    ENVIRONMENT_BASE_URL  — ClusterOps server URL (default: http://localhost:8000)
    HF_TOKEN              — HuggingFace token for gated models (optional for Qwen)

Usage:
    1. Start the server: uvicorn server.app:app --port 8000
    2. Run: python train_local.py
"""

import subprocess
import time
import os
import sys
import json
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
ENV_URL = os.getenv("ENVIRONMENT_BASE_URL", "http://localhost:8000")
# Using Qwen 0.5B because it is NOT gated and is extremely fast on CPUs
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN", None)

print(f"--- Starting Local Training Simulation ---")
print(f"Device: {DEVICE.upper()}")
if DEVICE == "cpu":
    print("WARNING: No GPU detected. Using Qwen-0.5B for maximum CPU speed.")


# --- Step 1: Start Server ---
def start_server():
    print("Checking if ClusterOps server is running...")
    try:
        if requests.get(f"{ENV_URL}/health", timeout=2).ok:
            print("Server already running.")
            return
    except Exception:
        pass

    print("Starting server in background...")
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)


# --- Step 2: Load Model ---
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, token=HF_TOKEN
).to(DEVICE)


# --- Step 3: Training Logic ---
SYSTEM_PROMPT = 'Output ONE JSON action only. {"action_type": "allocate", "job_id": "job_1", "node_id": 2}'


def run_episode(difficulty="easy", scenario="01_baseline"):
    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"difficulty": difficulty, "scenario": scenario},
        ).json()
    except Exception:
        print("Error: Could not connect to server. Is it running?")
        return 0

    total_reward = 0.0
    step = 0

    while not resp.get("done", False):
        obs = resp.get("observation")
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nStep {step}: {obs}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)

        text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Parse action
        try:
            action = json.loads(text[text.find("{"):text.rfind("}") + 1])
        except Exception:
            action = {"action_type": "wait"}

        resp = requests.post(f"{ENV_URL}/step", json=action).json()
        total_reward += resp.get("reward", 0)
        step += 1
        print(f"  Step {step}: Reward {resp.get('reward'):.1f} | Action: {action['action_type']}")

        if step > 10:
            break  # Small steps for local CPU run

    return total_reward


# --- Execution ---
start_server()
for ep in range(3):
    print(f"\nEpisode {ep+1} starting...")
    reward = run_episode()
    print(f"\nEpisode {ep+1} Complete. Total Reward: {reward:.1f}")
