"""
Unsloth-MLX Gradio GUI

A web-based interface for fine-tuning LLMs on Apple Silicon using Unsloth-MLX.
This provides an easy-to-use GUI for all Unsloth-MLX features.
"""

import gradio as gr
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datasets import load_dataset
from unsloth_mlx import (
    FastLanguageModel,
    SFTTrainer,
    SFTConfig,
    DPOTrainer,
    DPOConfig,
    ORPOTrainer,
    ORPOConfig,
    GRPOTrainer,
    GRPOConfig,
    KTOTrainer,
    SimPOTrainer,
    prepare_dataset,
    save_model_hf_format,
    export_to_gguf,
)


# Global state
class AppState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.training_in_progress = False


state = AppState()


def stable_copy_uploaded_file(upload_obj, dest_dir: str = "data/uploads") -> str:
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    if upload_obj is None:
        raise ValueError("No dataset file provided")

    if isinstance(upload_obj, str):
        src = upload_obj
    elif hasattr(upload_obj, "name") and isinstance(upload_obj.name, str):
        src = upload_obj.name
    elif isinstance(upload_obj, dict) and "name" in upload_obj:
        src = upload_obj["name"]
    else:
        raise ValueError(f"Unsupported dataset input type: {type(upload_obj)}")

    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"Uploaded dataset file not found at: {src}. "
            "Re-upload the dataset and start training again."
        )

    ts = int(time.time())
    dst = os.path.join(dest_dir, f"{ts}_{os.path.basename(src)}")
    shutil.copyfile(src, dst)
    return dst


def load_local_or_hub_dataset(dataset_path: str):
    if os.path.isfile(dataset_path):
        lower = dataset_path.lower()
        if lower.endswith(".jsonl") or lower.endswith(".json"):
            return load_dataset("json", data_files=dataset_path, split="train")
        if lower.endswith(".csv"):
            return load_dataset("csv", data_files=dataset_path, split="train")
        raise ValueError(f"Unsupported dataset file type: {dataset_path}")

    return load_dataset(dataset_path, split="train")


def _coerce_int(value, default: int):
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    if s == "":
        return default
    return int(float(s))


def _coerce_float(value, default: float):
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "":
        return default
    return float(s)


# ==================== Model Loading ====================


def load_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    hf_token: str = None,
):
    """Load a model from HuggingFace"""
    try:
        gr.Info(f"Loading model: {model_name}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            token=hf_token if hf_token else None,
        )

        state.model = model
        state.tokenizer = tokenizer
        state.model_name = model_name

        return gr.update(value=f"✓ Model loaded: {model_name}"), gr.update(interactive=True)
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        gr.Error(error_msg)
        return gr.update(value=error_msg), gr.update(interactive=False)


# ==================== Inference ====================


def generate_response(prompt: str, max_tokens: int, temperature: float, top_p: float):
    """Generate a response from the loaded model"""
    if state.model is None:
        return "Please load a model first!"

    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        # Enable inference mode
        FastLanguageModel.for_inference(state.model)

        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = state.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Generate response
        sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
        response = generate(
            state.model.model,
            state.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

        return response
    except Exception as e:
        return f"Error: {str(e)}"


def _first_selected_path(selection):
    if selection is None:
        return ""
    if isinstance(selection, list):
        if len(selection) == 0:
            return ""
        return str(selection[0])
    return str(selection)


def _choose_output_dir_from_explorer(selection, current_value: str):
    chosen = _first_selected_path(selection)
    if chosen == "":
        return str(current_value or ""), gr.update(visible=False)
    p = Path(chosen)
    if p.is_file():
        p = p.parent
    return str(p), gr.update(visible=False)


def _choose_output_file_in_dir_from_explorer(selection, current_value: str):
    chosen = _first_selected_path(selection)
    if chosen == "":
        return str(current_value or ""), gr.update(visible=False)
    p = Path(chosen)
    if p.is_file():
        p = p.parent
    name = Path(str(current_value or "model.gguf")).name
    if name.strip() == "":
        name = "model.gguf"
    return str(p / name), gr.update(visible=False)


# ==================== LoRA Configuration ====================


def apply_lora(r: int, lora_alpha: int, lora_dropout: float, target_modules: list):
    """Apply LoRA adapters to the model"""
    if state.model is None:
        return "Please load a model first!"

    try:
        state.model = FastLanguageModel.get_peft_model(
            state.model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )

        return f"✓ LoRA configured: rank={r}, alpha={lora_alpha}, modules={len(target_modules)}"
    except Exception as e:
        return f"Error configuring LoRA: {str(e)}"


def load_adapters(adapter_dir: str):
    if state.model is None:
        return "Error: Please load a model first!"

    adapter_dir = str(adapter_dir or "").strip()
    if adapter_dir == "":
        adapter_dir = "./adapters"

    adapter_path = Path(adapter_dir)
    weights_file = adapter_path / "adapters.safetensors"
    config_file = adapter_path / "adapter_config.json"
    if not weights_file.exists():
        return f"Error: Missing adapters.safetensors in {adapter_path}"

    try:
        mlx_model = state.model.model if hasattr(state.model, "model") else state.model

        if config_file.exists():
            from mlx_lm.tuner.utils import load_adapters as mlx_load_adapters

            mlx_load_adapters(mlx_model, str(adapter_path))
        else:
            from mlx_lm.tuner.utils import linear_to_lora_layers
            from safetensors import safe_open

            model_layers = None
            if hasattr(mlx_model, "layers"):
                model_layers = mlx_model.layers
            elif hasattr(mlx_model, "model") and hasattr(mlx_model.model, "layers"):
                model_layers = mlx_model.model.layers

            num_layers = len(model_layers) if model_layers is not None else 16

            guessed_keys = []
            guessed_rank = None
            with safe_open(str(weights_file), framework="np") as f:
                keys = list(f.keys())
                candidates = [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ]
                guessed_keys = [c for c in candidates if any(c in k for k in keys)]

                for k in keys:
                    try:
                        shape = f.get_tensor(k).shape
                    except Exception:
                        continue
                    if len(shape) == 2:
                        guessed_rank = int(min(shape))
                        break

            if not guessed_keys:
                guessed_keys = [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                ]

            if guessed_rank is None:
                guessed_rank = 16

            lora_parameters = {
                "rank": guessed_rank,
                "scale": 2.0,
                "dropout": 0.0,
                "keys": guessed_keys,
            }

            linear_to_lora_layers(
                mlx_model,
                num_layers,
                lora_parameters,
                use_dora=False,
            )
            mlx_model.load_weights(str(weights_file), strict=False)

            try:
                cfg = {
                    "fine_tune_type": "lora",
                    "num_layers": num_layers,
                    "lora_parameters": lora_parameters,
                }
                adapter_path.mkdir(parents=True, exist_ok=True)
                config_file.write_text(json.dumps(cfg, indent=2))
            except Exception:
                pass

        if hasattr(state.model, "set_adapter_path"):
            state.model.set_adapter_path(str(adapter_path))
        return f"✓ Adapters loaded from {adapter_path}"
    except Exception as e:
        return f"Error loading adapters: {str(e)}"


# ==================== SFT Training ====================


def run_sft_training(
    dataset_file,
    output_dir: str,
    num_train_epochs: int,
    learning_rate: float,
    batch_size: int,
    max_steps: int,
    progress=gr.Progress(),
):
    """Run SFT training"""
    if state.model is None:
        return "Error: Please load a model first!"

    if dataset_file is None:
        return "Error: Please upload a training dataset!"

    try:
        num_train_epochs = max(1, _coerce_int(num_train_epochs, 3))
        learning_rate = _coerce_float(learning_rate, 2e-4)
        batch_size = max(1, _coerce_int(batch_size, 4))
        max_steps = max(0, _coerce_int(max_steps, 0))

        progress(0, desc="Preparing dataset...")

        dataset_path = stable_copy_uploaded_file(dataset_file)
        print(f"Loading dataset '{dataset_path}'...")
        dataset = load_local_or_hub_dataset(dataset_path)

        progress(0.2, desc="Configuring trainer...")

        # Configure training
        trainer = SFTTrainer(
            model=state.model,
            train_dataset=dataset,
            tokenizer=state.tokenizer,
            args=SFTConfig(
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                max_steps=max_steps,
            ),
        )

        progress(0.4, desc="Starting training...")

        # Train
        trainer.train()

        progress(1.0, desc="Training complete!")

        return f"✓ Training complete! Model saved to {output_dir}"
    except Exception as e:
        return f"Error during training: {str(e)}"


# ==================== RL Training ====================


def run_rl_training(
    method: str,
    dataset_file,
    output_dir: str,
    learning_rate: float,
    max_steps: int,
    beta: float,
    progress=gr.Progress(),
):
    """Run RL training (DPO/ORPO/GRPO/KTO/SimPO)"""
    if state.model is None:
        return "Error: Please load a model first!"

    if dataset_file is None:
        return "Error: Please upload a preference dataset!"

    try:
        learning_rate = _coerce_float(learning_rate, 5e-7)
        max_steps = max(1, _coerce_int(max_steps, 100))
        beta = _coerce_float(beta, 0.1)

        progress(0, desc="Preparing dataset...")

        # Load preference dataset
        from unsloth_mlx import prepare_preference_dataset

        dataset_path = stable_copy_uploaded_file(dataset_file)
        print(f"Loading dataset '{dataset_path}'...")
        raw_dataset = load_local_or_hub_dataset(dataset_path)

        format_type = method.lower() if method.lower() in ["dpo", "orpo", "grpo"] else "dpo"
        dataset = prepare_preference_dataset(raw_dataset, state.tokenizer, format_type)

        progress(0.2, desc="Configuring trainer...")

        # Select trainer based on method
        if method == "DPO":
            config = DPOConfig(
                beta=beta,
                learning_rate=learning_rate,
                max_steps=max_steps,
                output_dir=output_dir,
            )
            trainer = DPOTrainer(
                model=state.model,
                train_dataset=dataset,
                tokenizer=state.tokenizer,
                args=config,
            )
        elif method == "ORPO":
            config = ORPOConfig(
                beta=beta,
                learning_rate=learning_rate,
                max_steps=max_steps,
                output_dir=output_dir,
            )
            trainer = ORPOTrainer(
                model=state.model,
                train_dataset=dataset,
                tokenizer=state.tokenizer,
                args=config,
            )
        elif method == "GRPO":
            config = GRPOConfig(
                learning_rate=learning_rate,
                max_steps=max_steps,
                output_dir=output_dir,
            )
            trainer = GRPOTrainer(
                model=state.model,
                train_dataset=dataset,
                tokenizer=state.tokenizer,
                args=config,
            )
        elif method == "KTO":
            trainer = KTOTrainer(
                model=state.model,
                train_dataset=dataset,
                tokenizer=state.tokenizer,
                learning_rate=learning_rate,
                max_steps=max_steps,
                output_dir=output_dir,
            )
        elif method == "SimPO":
            trainer = SimPOTrainer(
                model=state.model,
                train_dataset=dataset,
                tokenizer=state.tokenizer,
                learning_rate=learning_rate,
                max_steps=max_steps,
                output_dir=output_dir,
            )

        progress(0.4, desc="Starting training...")

        # Train
        trainer.train()

        progress(1.0, desc="Training complete!")

        return f"✓ {method} training complete! Model saved to {output_dir}"
    except Exception as e:
        return f"Error during training: {str(e)}"


# ==================== Export ====================


def save_adapters(output_path: str):
    """Save LoRA adapters"""
    if state.model is None:
        return "Error: Please load a model first!"

    try:
        state.model.save_pretrained(output_path)
        return f"✓ Adapters saved to {output_path}"
    except Exception as e:
        return f"Error: {str(e)}"


def save_merged_model(output_path: str):
    """Save merged model in HuggingFace format"""
    if state.model is None:
        return "Error: Please load a model first!"

    try:
        save_model_hf_format(state.model, state.tokenizer, output_path, base_model_name=state.model_name)
        return f"✓ Merged model saved to {output_path}"
    except Exception as e:
        return f"Error: {str(e)}"


def export_gguf(output_path: str, quantization: str):
    """Export model to GGUF format"""
    if state.model is None:
        return "Error: Please load a model first!"

    try:
        adapter_path = None
        if hasattr(state.model, "get_adapter_path"):
            try:
                adapter_path = state.model.get_adapter_path()
            except Exception:
                adapter_path = None

        model_path = state.model_name
        if not model_path:
            return "Error: Missing base model name/path. Please reload the model."

        export_to_gguf(
            model_path=model_path,
            output_path=output_path,
            quantization=quantization,
            adapter_path=adapter_path,
        )
        return f"✓ GGUF exported to {output_path} ({quantization})"
    except Exception as e:
        return f"Error: {str(e)}"


# ==================== Dataset Format Helper ====================


def show_dataset_format(format_type: str):
    """Show example dataset formats"""
    if format_type == "SFT (Instruction Tuning)":
        return json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": "What is Python?"},
                        {"role": "assistant", "content": "Python is a programming language..."},
                    ]
                }
            ],
            indent=2,
        )
    elif format_type == "DPO (Preference)":
        return json.dumps(
            [
                {
                    "prompt": "Explain machine learning",
                    "chosen": "Machine learning is a branch of AI...",
                    "rejected": "idk it's like computers doing stuff",
                }
            ],
            indent=2,
        )
    return ""


# ==================== Build Interface ====================


def build_ui():
    """Build the Gradio interface"""

    with gr.Blocks(
        title="Unsloth-MLX GUI",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: 0 auto; }
        """,
    ) as demo:
        gr.Markdown("""
        # 🚀 Unsloth-MLX GUI
        
        Fine-tune LLMs on your Apple Silicon Mac with a user-friendly interface.
        
        **Platform:** Apple Silicon (M1/M2/M3/M4/M5) | **Framework:** MLX | **API:** Compatible with Unsloth
        """)

        with gr.Tabs():
            # ========== Tab 1: Model Loading ==========
            with gr.Tab("📥 Load Model"):
                gr.Markdown("### Load a model from HuggingFace")

                with gr.Row():
                    model_name = gr.Textbox(
                        label="Model Name",
                        value="mlx-community/Llama-3.2-1B-Instruct-4bit",
                        placeholder="e.g., mlx-community/Llama-3.2-1B-Instruct-4bit",
                        info="Enter HuggingFace model name or path",
                    )
                    hf_token = gr.Textbox(
                        label="HuggingFace Token (Optional)",
                        type="password",
                        placeholder="For gated models like Llama",
                    )

                with gr.Row():
                    max_seq_length = gr.Slider(
                        label="Max Sequence Length", minimum=512, maximum=8192, value=2048, step=512
                    )

                with gr.Row():
                    load_in_4bit = gr.Checkbox(label="Load in 4-bit (recommended)", value=True)
                    load_in_8bit = gr.Checkbox(label="Load in 8-bit", value=False)

                load_btn = gr.Button("Load Model", variant="primary", size="lg")
                load_status = gr.Textbox(label="Status", value="No model loaded", interactive=False)

                load_btn.click(
                    fn=load_model,
                    inputs=[model_name, max_seq_length, load_in_4bit, load_in_8bit, hf_token],
                    outputs=[load_status, load_btn],
                )

                with gr.Row():
                    adapters_dir = gr.Textbox(
                        label="Adapters Folder",
                        value="./adapters",
                        placeholder="./adapters",
                        scale=4,
                    )
                    load_adapters_btn = gr.Button("Load Adapters", variant="secondary", scale=1)

                adapters_status = gr.Textbox(
                    label="Adapters Status",
                    value="",
                    interactive=False,
                )

                load_adapters_btn.click(
                    fn=load_adapters,
                    inputs=[adapters_dir],
                    outputs=[adapters_status],
                )

                gr.Markdown("""
                **Recommended Models:**
                - 1B: `mlx-community/Llama-3.2-1B-Instruct-4bit` (16GB RAM)
                - 3B: `mlx-community/Llama-3.2-3B-Instruct-4bit` (32GB RAM)
                - 7B: `mlx-community/Llama-3.2-7B-Instruct-4bit` (48GB+ RAM)
                """)

            # ========== Tab 2: Inference ==========
            with gr.Tab("💬 Chat"):
                gr.Markdown("### Chat with your loaded model")

                chatbot = gr.Chatbot(label="Conversation", height=400)

                with gr.Row():
                    prompt = gr.Textbox(
                        label="Your Message", placeholder="Type your message here...", scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Accordion("Generation Parameters", open=False):
                    with gr.Row():
                        max_tokens = gr.Slider(
                            label="Max Tokens", minimum=32, maximum=2048, value=256, step=32
                        )
                        temperature = gr.Slider(
                            label="Temperature", minimum=0.1, maximum=2.0, value=0.7, step=0.1
                        )
                        top_p = gr.Slider(
                            label="Top P", minimum=0.1, maximum=1.0, value=0.9, step=0.05
                        )

                def _normalize_chat_history(history):
                    if history is None:
                        return []

                    normalized = []
                    for item in history:
                        if isinstance(item, dict) and "role" in item and "content" in item:
                            normalized.append(item)
                            continue

                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            user, assistant = item
                            if user not in (None, ""):
                                normalized.append({"role": "user", "content": str(user)})
                            if assistant not in (None, ""):
                                normalized.append({"role": "assistant", "content": str(assistant)})
                            continue

                    return normalized

                def chat_fn(message, history, max_tokens, temperature, top_p):
                    history = _normalize_chat_history(history)
                    user_text = str(message).strip()
                    if user_text == "":
                        return history, ""

                    history.append({"role": "user", "content": user_text})

                    reply = generate_response(user_text, max_tokens, temperature, top_p)
                    history.append({"role": "assistant", "content": reply})
                    return history, ""

                send_btn.click(
                    fn=chat_fn,
                    inputs=[prompt, chatbot, max_tokens, temperature, top_p],
                    outputs=[chatbot, prompt],
                )

                prompt.submit(
                    fn=chat_fn,
                    inputs=[prompt, chatbot, max_tokens, temperature, top_p],
                    outputs=[chatbot, prompt],
                )

            # ========== Tab 3: LoRA Config ==========
            with gr.Tab("🔧 LoRA Configuration"):
                gr.Markdown("### Configure LoRA adapters for efficient fine-tuning")

                with gr.Row():
                    lora_r = gr.Slider(
                        label="LoRA Rank (r)",
                        minimum=4,
                        maximum=64,
                        value=16,
                        step=4,
                        info="Higher = more parameters, better quality",
                    )
                    lora_alpha = gr.Slider(
                        label="LoRA Alpha",
                        minimum=8,
                        maximum=128,
                        value=32,
                        step=8,
                        info="Typically 2x rank",
                    )
                    lora_dropout = gr.Slider(
                        label="LoRA Dropout", minimum=0.0, maximum=0.2, value=0.05, step=0.01
                    )

                target_modules = gr.CheckboxGroup(
                    label="Target Modules",
                    choices=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    value=["q_proj", "k_proj", "v_proj", "o_proj"],
                    info="Which layers to apply LoRA to",
                )

                apply_lora_btn = gr.Button("Apply LoRA", variant="primary")
                lora_status = gr.Textbox(label="Status", value="", interactive=False)

                apply_lora_btn.click(
                    fn=apply_lora,
                    inputs=[lora_r, lora_alpha, lora_dropout, target_modules],
                    outputs=[lora_status],
                )

            # ========== Tab 4: SFT Training ==========
            with gr.Tab("📚 SFT Training"):
                gr.Markdown("### Supervised Fine-Tuning for instruction following")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Upload Dataset")
                        dataset_sft = gr.File(
                            label="Training Dataset (JSONL) — Upload your training data",
                            type="filepath",
                            file_types=[".jsonl"],
                        )

                        format_examples_sft = gr.Radio(
                            label="Show Dataset Format Example",
                            choices=["SFT (Instruction Tuning)", "DPO (Preference)"],
                            value="SFT (Instruction Tuning)",
                        )
                        format_display_sft = gr.Code(
                            label="Dataset Format",
                            language="json",
                            value=show_dataset_format("SFT (Instruction Tuning)"),
                        )

                        format_examples_sft.change(
                            fn=show_dataset_format,
                            inputs=[format_examples_sft],
                            outputs=[format_display_sft],
                        )

                    with gr.Column():
                        gr.Markdown("#### Training Parameters")
                        output_dir_sft = gr.Textbox(
                            label="Output Directory",
                            value="./sft_output",
                            placeholder="./sft_output",
                        )
                        num_train_epochs = gr.Number(
                            label="Training Epochs", value=3, minimum=1, maximum=100
                        )
                        learning_rate_sft = gr.Textbox(
                            label="Learning Rate", value="2e-4", placeholder="2e-4"
                        )
                        batch_size_sft = gr.Slider(
                            label="Batch Size", minimum=1, maximum=16, value=4, step=1
                        )
                        max_steps_sft = gr.Number(
                            label="Max Steps (0 = full epochs)", value=0, minimum=0
                        )

                train_sft_btn = gr.Button("Start SFT Training", variant="primary", size="lg")
                training_output_sft = gr.Textbox(
                    label="Training Output", value="", interactive=False
                )

                train_sft_btn.click(
                    fn=run_sft_training,
                    inputs=[
                        dataset_sft,
                        output_dir_sft,
                        num_train_epochs,
                        learning_rate_sft,
                        batch_size_sft,
                        max_steps_sft,
                    ],
                    outputs=[training_output_sft],
                )

            # ========== Tab 5: RL Training ==========
            with gr.Tab("🎯 RL Training"):
                gr.Markdown("### Reinforcement Learning Training Methods")

                with gr.Row():
                    method = gr.Radio(
                        label="Training Method",
                        choices=["DPO", "ORPO", "GRPO", "KTO", "SimPO"],
                        value="DPO",
                        info="Select RL training method",
                    )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Upload Dataset")
                        dataset_rl = gr.File(
                            label="Preference Dataset (JSONL) — Upload preference data (chosen/rejected)",
                            type="filepath",
                            file_types=[".jsonl"],
                        )

                        format_examples_rl = gr.Radio(
                            label="Show Dataset Format Example",
                            choices=["SFT (Instruction Tuning)", "DPO (Preference)"],
                            value="DPO (Preference)",
                        )
                        format_display_rl = gr.Code(
                            label="Dataset Format",
                            language="json",
                            value=show_dataset_format("DPO (Preference)"),
                        )

                        format_examples_rl.change(
                            fn=show_dataset_format,
                            inputs=[format_examples_rl],
                            outputs=[format_display_rl],
                        )

                    with gr.Column():
                        gr.Markdown("#### Training Parameters")
                        output_dir_rl = gr.Textbox(
                            label="Output Directory", value="./rl_output", placeholder="./rl_output"
                        )
                        learning_rate_rl = gr.Textbox(
                            label="Learning Rate", value="5e-7", placeholder="5e-7"
                        )
                        max_steps_rl = gr.Number(label="Max Steps", value=100, minimum=1)
                        beta = gr.Slider(
                            label="Beta (KL Penalty)",
                            minimum=0.01,
                            maximum=1.0,
                            value=0.1,
                            step=0.01,
                            info="For DPO/ORPO only",
                        )

                train_rl_btn = gr.Button("Start RL Training", variant="primary", size="lg")
                training_output_rl = gr.Textbox(
                    label="Training Output", value="", interactive=False
                )

                train_rl_btn.click(
                    fn=run_rl_training,
                    inputs=[
                        method,
                        dataset_rl,
                        output_dir_rl,
                        learning_rate_rl,
                        max_steps_rl,
                        beta,
                    ],
                    outputs=[training_output_rl],
                )

                gr.Markdown("""
                **Training Methods:**
                - **DPO**: Direct Preference Optimization - trains on chosen/rejected pairs
                - **ORPO**: Odds Ratio Preference Optimization - combines SFT + preference
                - **GRPO**: Group Relative Policy Optimization - for reasoning tasks (DeepSeek R1)
                - **KTO**: Kahneman-Tversky Optimization - for unpaired preference data
                - **SimPO**: Simple Preference Optimization - simplified DPO
                """)

            # ========== Tab 6: Export ==========
            with gr.Tab("💾 Export"):
                gr.Markdown("### Save and export your fine-tuned model")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Save Adapters")
                        with gr.Row():
                            adapter_output = gr.Textbox(
                                label="Output Path", value="./adapters", placeholder="./adapters"
                            )
                            browse_adapters_btn = gr.Button("Browse…")
                        adapters_browse_panel = gr.Column(visible=False)
                        with adapters_browse_panel:
                            adapters_picker = gr.FileExplorer(
                                root_dir=str(Path.home()),
                                file_count="single",
                                ignore_glob="**/.*",
                                height=260,
                                label=None,
                            )
                            use_adapters_picker_btn = gr.Button("Use Selection")
                        save_adapters_btn = gr.Button("Save LoRA Adapters", variant="primary")
                        adapter_status = gr.Textbox(label="Status", value="", interactive=False)

                    with gr.Column():
                        gr.Markdown("#### Save Merged Model (HuggingFace Format)")
                        with gr.Row():
                            merged_output = gr.Textbox(
                                label="Output Path",
                                value="./merged_model",
                                placeholder="./merged_model",
                            )
                            browse_merged_btn = gr.Button("Browse…")
                        merged_browse_panel = gr.Column(visible=False)
                        with merged_browse_panel:
                            merged_picker = gr.FileExplorer(
                                root_dir=str(Path.home()),
                                file_count="single",
                                ignore_glob="**/.*",
                                height=260,
                                label=None,
                            )
                            use_merged_picker_btn = gr.Button("Use Selection")
                        save_merged_btn = gr.Button("Save Merged Model", variant="primary")
                        merged_status = gr.Textbox(label="Status", value="", interactive=False)

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Export to GGUF")
                        with gr.Row():
                            gguf_output = gr.Textbox(
                                label="Output Path", value="./model.gguf", placeholder="./model.gguf"
                            )
                            browse_gguf_btn = gr.Button("Browse…")
                        gguf_browse_panel = gr.Column(visible=False)
                        with gguf_browse_panel:
                            gguf_picker = gr.FileExplorer(
                                root_dir=str(Path.home()),
                                file_count="single",
                                ignore_glob="**/.*",
                                height=260,
                                label=None,
                            )
                            use_gguf_picker_btn = gr.Button("Use Selection")
                        quantization = gr.Dropdown(
                            label="Quantization Method",
                            choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                            value="q4_k_m",
                            info="Lower = smaller, faster; Higher = better quality",
                        )
                        export_gguf_btn = gr.Button("Export to GGUF", variant="primary")
                        gguf_status = gr.Textbox(label="Status", value="", interactive=False)

                browse_adapters_btn.click(lambda: gr.update(visible=True), outputs=[adapters_browse_panel])
                browse_merged_btn.click(lambda: gr.update(visible=True), outputs=[merged_browse_panel])
                browse_gguf_btn.click(lambda: gr.update(visible=True), outputs=[gguf_browse_panel])

                use_adapters_picker_btn.click(
                    fn=_choose_output_dir_from_explorer,
                    inputs=[adapters_picker, adapter_output],
                    outputs=[adapter_output, adapters_browse_panel],
                )
                use_merged_picker_btn.click(
                    fn=_choose_output_dir_from_explorer,
                    inputs=[merged_picker, merged_output],
                    outputs=[merged_output, merged_browse_panel],
                )
                use_gguf_picker_btn.click(
                    fn=_choose_output_file_in_dir_from_explorer,
                    inputs=[gguf_picker, gguf_output],
                    outputs=[gguf_output, gguf_browse_panel],
                )

                save_adapters_btn.click(
                    fn=save_adapters, inputs=[adapter_output], outputs=[adapter_status]
                )

                save_merged_btn.click(
                    fn=save_merged_model, inputs=[merged_output], outputs=[merged_status]
                )

                export_gguf_btn.click(
                    fn=export_gguf, inputs=[gguf_output, quantization], outputs=[gguf_status]
                )

                gr.Markdown("""
                **Export Options:**
                - **Adapters**: Just the LoRA weights (small file, fast to save)
                - **Merged Model**: Full model in HuggingFace format (compatible with transformers, vLLM)
                - **GGUF**: For llama.cpp, Ollama, GPT4All (CPU inference)
                """)

            # ========== Tab 7: Documentation ==========
            with gr.Tab("📖 Documentation"):
                gr.Markdown("""
                ### Unsloth-MLX GUI Documentation
                
                #### Getting Started
                1. Go to **Load Model** tab and select a model from HuggingFace
                2. Click **Load Model** and wait for it to load
                3. Configure **LoRA** if you want to fine-tune
                4. Go to **Chat** to test the model, or **Training** to fine-tune
                5. **Export** your model when done
                
                #### Dataset Formats
                
                **SFT (Instruction Tuning):**
                ```json
                {"messages": [
                    {"role": "user", "content": "What is Python?"},
                    {"role": "assistant", "content": "Python is a programming language..."}
                ]}
                ```
                
                **DPO (Preference):**
                ```json
                {
                    "prompt": "Explain machine learning",
                    "chosen": "Machine learning is a branch of AI...",
                    "rejected": "idk it's like computers doing stuff"
                }
                ```
                
                #### Training Methods
                
                - **SFT**: Supervised Fine-Tuning - teaches instruction following
                - **DPO**: Direct Preference Optimization - improves response quality
                - **ORPO**: Odds Ratio Preference Optimization - combines SFT + preference
                - **GRPO**: Group Relative Policy Optimization - for reasoning tasks
                - **KTO**: Kahneman-Tversky Optimization - for unpaired data
                - **SimPO**: Simple Preference Optimization - simplified DPO
                
                #### Tips
                
                - Start with a small model (1B-3B) for testing
                - Use 4-bit quantization to save memory
                - Use low learning rates (2e-4 to 5e-7) for fine-tuning
                - Train for 3-10 epochs for best results
                - Export to GGUF for easy deployment
                
                #### Hardware Recommendations
                
                - **16GB RAM**: 1B-3B models with 4-bit
                - **32GB RAM**: 7B models with 4-bit
                - **48GB+ RAM**: 7B-13B models with 4-bit or 8-bit
                
                For more details, visit the [Unsloth-MLX GitHub](https://github.com/dax8it/unsloth-mlx)
                """)

        gr.Markdown("""
        ---
        
        **Unsloth-MLX GUI** - Built with Gradio for Apple Silicon
        
        Fine-tune LLMs locally on your Mac. Prototype now, scale to cloud later!
        """)

    return demo


# ==================== Main ====================


def main():
    """Main entry point"""
    demo = build_ui()

    print("=" * 70)
    print("Unsloth-MLX GUI")
    print("=" * 70)
    print("Starting Gradio server...")
    print("Open your browser to the URL shown below")
    print("=" * 70)

    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    main()
