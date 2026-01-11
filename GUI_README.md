# Unsloth-MLX GUI

A user-friendly web interface for fine-tuning LLMs on Apple Silicon using Unsloth-MLX.

## Features

- **Model Loading**: Load any HuggingFace model with one click
- **Chat Interface**: Test your models with a built-in chat UI
- **LoRA Configuration**: Configure parameter-efficient fine-tuning
- **SFT Training**: Supervised fine-tuning for instruction following
- **RL Training**: Multiple RL methods (DPO, ORPO, GRPO, KTO, SimPO)
- **Export Options**: Save adapters, merge models, export to GGUF

## Installation

```bash
# Create & activate a virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate

# Install from source
git clone https://github.com/dax8it/unsloth-mlx.git
cd unsloth-mlx
pip install -e .
```

## Running the GUI

```bash
python gui.py
```

Or:

```bash
./launch_gui.sh
```

The GUI will open at `http://127.0.0.1:7860` in your browser.

## Quick Start

1. **Load a Model**: Go to the "Load Model" tab and select a model
   - Recommended: `mlx-community/Llama-3.2-1B-Instruct-4bit`
   - Click "Load Model"

2. **(Optional) Load Adapters**: If you trained a LoRA already, load it here
   - Set the adapters folder (default: `./adapters`)
   - Click "Load Adapters"

3. **Test Model**: Go to "Chat" tab
   - Type a message and chat with your model

4. **Fine-tune**: Go to "SFT Training" or "RL Training"
   - Upload your dataset (JSONL format)
   - If your SFT dataset isn't already in `{"messages": [...]}` format, click **Convert to messages JSONL** (SFT tab)
   - Configure training parameters
   - Click "Start Training"

5. **Export**: Go to "Export" tab
   - Save adapters, save a merged model, or export to GGUF

## Export Notes

- **Browse buttons**: The Export tab includes "Browse…" buttons that open a file explorer (server-side) and auto-fill output paths.
- **Save LoRA Adapters**: Saves `adapters.safetensors` + `adapter_config.json` into a folder you choose.
- **Save Merged Model (MLX folder)**:
  - If adapters are loaded, the GUI will fuse them into the base weights using `mlx_lm.fuse` so external tools (e.g. LM Studio MLX) can load the folder.
  - The exporter also attempts to include `config.json` for better compatibility.
- **Export to GGUF**:
  - GGUF export is only supported by `mlx_lm` for model families: `llama`, `mistral`, `mixtral`.
  - Some models (e.g. `model_type: lfm2`) cannot be exported to GGUF with `mlx_lm`.

## LEAP GGUF Export (Liquid AI)

If you're fine-tuning a LEAP-supported base architecture (LFM2 / LFM2-VL / Qwen), you can bundle an iOS-ready GGUF using Liquid AI's `leap-bundle` service.

1. Install the CLI:
```bash
pip install leap-bundle
```

2. Authenticate (optional if you paste an API key in the GUI):
```bash
leap-bundle login <api-key>
```

3. In the GUI **Export** tab:
- Run **Save Merged Model** (produces the checkpoint folder)
- In **LEAP GGUF Bundling**, click **Validate Directory**
- Click **Create Bundle Request** and wait for processing
- Click **Check Status** until the request is completed
- Click **Download GGUF** to save the `.gguf` to your chosen folder

## LM Studio

LM Studio can load MLX models, but it generally expects a "standard" MLX model folder (no LoRA parameters like `lora_a`/`lora_b`).

- Use **Save Merged Model** after loading adapters to generate a fused MLX folder.
- Then point LM Studio at that fused folder.

## Dataset Formats

### SFT (Instruction Tuning)
```json
{"messages": [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
]}
```

If your SFT dataset is not in `messages` format, the GUI can convert common formats into `messages` JSONL:
- `conversations` (ShareGPT-style `from/value`)
- Alpaca-style: `instruction` + optional `input` + `output`
- Prompt/completion: `prompt` + `completion` (or `response`)

Converted datasets are written to `data/converted/` and the GUI auto-fills the dataset path to the converted file.

### DPO (Preference)
```json
{
    "prompt": "Explain machine learning",
    "chosen": "Machine learning is a branch of AI...",
    "rejected": "idk it's like computers doing stuff"
}
```

## Hardware Requirements

- **16GB RAM**: 1B-3B models with 4-bit
- **32GB RAM**: 7B models with 4-bit
- **48GB+ RAM**: 7B-13B models with 4-bit or 8-bit

## Tips

- Start with small models for testing
- Use 4-bit quantization to save memory
- Use low learning rates (2e-4 to 5e-7)
- Train for 3-10 epochs for best results
- Export to GGUF for easy deployment with Ollama or llama.cpp (LLaMA/Mistral/Mixtral-family models only)
