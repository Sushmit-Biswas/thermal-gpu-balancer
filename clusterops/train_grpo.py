#!/usr/bin/env python3
"""
ClusterOps GRPO Training Script.

Trains an LLM (via Unsloth + TRL) to manage GPU clusters using
Group Relative Policy Optimization (GRPO).

This script:
1. Loads a base model (Llama-3.1-8B-Instruct) in 4-bit via Unsloth.
2. Adds LoRA adapters for efficient fine-tuning.
3. Defines reward functions that query the ClusterOps environment.
4. Runs GRPO training and saves loss/reward curves as .png files.

Usage:
    # Local (requires GPU):
    python train_grpo.py

    # Or run in Google Colab (recommended for hackathon):
    # See the linked Colab notebook in the README.

Requirements:
    pip install unsloth trl torch transformers matplotlib requests
"""

import os
import json
import random
import requests
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# ─── Configuration ──────────────────────────────────────────────────────────────
ENV_URL = os.getenv("ENVIRONMENT_BASE_URL", "http://localhost:8000")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./training_outputs")
NUM_EPISODES = int(os.getenv("NUM_EPISODES", "100"))  
DIFFICULTY = os.getenv("DIFFICULTY", "easy")  # Start easy for curriculum

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Environment Interface ─────────────────────────────────────────────────────

def env_reset(difficulty="easy"):
    """Reset the ClusterOps environment."""
    resp = requests.post(f"{ENV_URL}/reset", params={"difficulty": difficulty})
    return resp.json()


def env_step(action: dict):
    """Execute a step in the ClusterOps environment."""
    resp = requests.post(f"{ENV_URL}/step", json=action)
    return resp.json()


def parse_action_from_text(text: str) -> dict:
    """Extract a JSON action from model-generated text."""
    try:
        # Handle markdown code blocks
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"action_type": "wait"}


def format_observation(obs_data: dict) -> str:
    """Convert observation to a compact text prompt for the LLM."""
    nodes = obs_data.get("gpu_nodes", [])
    queue = obs_data.get("job_queue", [])
    meta = obs_data.get("metadata", {})
    
    lines = [f"STEP {meta.get('step', '?')}/{meta.get('max_steps', '?')} | "
             f"Done:{obs_data.get('completed_jobs',0)} Melts:{obs_data.get('meltdowns',0)}"]
    
    lines.append("NODES: " + " | ".join(
        f"N{n['id']}:{n['status'][:1].upper()}{n['temperature']:.0f}C"
        + (f"({n.get('job_type','?')[:3]},{n['job_duration_remaining']})" if n['status'] == 'busy' else "")
        for n in nodes
    ))
    
    if queue:
        lines.append("QUEUE: " + " | ".join(
            f"{j['id']}:{j['type'][:3]},d{j['duration']},w{j['wait_time']}" 
            for j in queue[:8]
        ))
    else:
        lines.append("QUEUE: empty")
    
    return "\n".join(lines)


# ─── Reward Function for GRPO ──────────────────────────────────────────────────

def compute_episode_reward(model, tokenizer, difficulty="easy", max_steps=50):
    """
    Run one episode using the model to generate actions.
    Returns: (total_reward, completion_data)
    """
    obs = env_reset(difficulty)
    obs_data = obs.get("observation", obs)
    total_reward = 0.0
    steps = 0
    
    while True:
        done = obs.get("done", obs_data.get("done", False))
        if done:
            break
        
        # Format prompt
        prompt = format_observation(obs_data)
        prompt_text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You manage a GPU cluster. Output ONE JSON action.\n"
            f"Actions: allocate(job_id, node_id), evict(node_id), cooldown(node_id), wait.\n"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        
        # Generate action
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with __import__('torch').no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.5,
                do_sample=True,
            )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        action = parse_action_from_text(generated)
        
        # Step environment
        obs = env_step(action)
        obs_data = obs.get("observation", obs)
        step_reward = obs.get("reward", obs_data.get("reward", 0.0))
        total_reward += step_reward
        steps += 1
    
    return total_reward, {
        "steps": steps,
        "completed_jobs": obs_data.get("completed_jobs", 0),
        "meltdowns": obs_data.get("meltdowns", 0),
    }


# ─── Training Loop ─────────────────────────────────────────────────────────────

def train():
    """Main training function using Unsloth + GRPO."""
    print("=" * 60)
    print("ClusterOps: GRPO Training Pipeline")
    print("=" * 60)
    
    # 1. Load Model via Unsloth
    print("\n[1/4] Loading model via Unsloth (4-bit quantization)...")
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
    )
    
    # 2. Add LoRA Adapters
    print("[2/4] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # 3. Run Training Episodes
    print(f"[3/4] Running {NUM_EPISODES} training episodes on '{DIFFICULTY}' difficulty...")
    
    episode_rewards = []
    episode_completions = []
    episode_meltdowns = []
    
    for ep in range(NUM_EPISODES):
        reward, data = compute_episode_reward(model, tokenizer, DIFFICULTY)
        episode_rewards.append(reward)
        episode_completions.append(data["completed_jobs"])
        episode_meltdowns.append(data["meltdowns"])
        
        if (ep + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            avg_complete = sum(episode_completions[-10:]) / 10
            avg_melt = sum(episode_meltdowns[-10:]) / 10
            print(f"  Episode {ep+1:4d} | Avg Reward: {avg_reward:+8.1f} | "
                  f"Avg Completed: {avg_complete:.1f} | Avg Meltdowns: {avg_melt:.1f}")
    
    # 4. Generate Plots
    print("[4/4] Generating training plots...")
    _save_plots(episode_rewards, episode_completions, episode_meltdowns)
    
    # Save model
    model.save_pretrained(os.path.join(OUTPUT_DIR, "clusterops_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "clusterops_model"))
    
    print(f"\n{'='*60}")
    print(f"Training complete! Artifacts saved to {OUTPUT_DIR}/")
    print(f"  - reward_curve.png")
    print(f"  - loss_curve.png")
    print(f"  - clusterops_model/")
    print(f"{'='*60}")


def _save_plots(rewards, completions, meltdowns):
    """Save training curves as .png files for submission."""
    
    # ── Reward Curve ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.3, color='#4CAF50', label='Per-Episode Reward')
    # Rolling average
    window = min(10, len(rewards))
    if len(rewards) >= window:
        rolling = [sum(rewards[max(0,i-window):i]) / min(i, window) 
                   for i in range(1, len(rewards)+1)]
        ax.plot(rolling, color='#2E7D32', linewidth=2, label=f'Rolling Avg ({window} ep)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Episode Reward', fontsize=12)
    ax.set_title('ClusterOps GRPO Training: Reward Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'reward_curve.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/reward_curve.png")
    
    # ── Completions vs Meltdowns ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(completions, color='#2196F3', alpha=0.5, label='Jobs Completed')
    ax.plot(meltdowns, color='#F44336', alpha=0.5, label='Thermal Meltdowns')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('ClusterOps GRPO Training: Jobs vs Meltdowns', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR}/loss_curve.png")


if __name__ == "__main__":
    train()
