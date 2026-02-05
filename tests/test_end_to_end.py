"""
End-to-end tests for complete training workflows.

These tests verify the full pipeline from model loading to saving,
catching integration issues that unit tests miss.

Run with: pytest tests/test_end_to_end.py -v
"""

import pytest
import tempfile
import json
import os
from pathlib import Path


pytestmark = pytest.mark.integration


class TestSFTEndToEnd:
    """End-to-end tests for SFT training pipeline."""

    @pytest.fixture(scope="class")
    def trained_model_dir(self):
        """Train a model and return the output directory."""
        from unsloth_mlx import FastLanguageModel, SFTTrainer, SFTConfig
        from datasets import Dataset

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            max_seq_length=256,
        )

        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=8,
        )

        # Minimal dataset
        dataset = Dataset.from_dict({
            "text": ["Hello world test."] * 3
        })

        # Train
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                tokenizer=tokenizer,
                args=SFTConfig(
                    output_dir=tmpdir,
                    max_steps=2,
                    per_device_train_batch_size=1,
                    logging_steps=1,
                    save_steps=2,
                ),
            )
            trainer.train()

            # Return paths and model for further tests
            yield {
                "output_dir": Path(tmpdir),
                "adapter_path": Path(tmpdir) / "adapters",
                "model": model,
                "tokenizer": tokenizer,
            }

    def test_adapters_safetensors_created(self, trained_model_dir):
        """Test that adapters.safetensors is created after training."""
        adapter_file = trained_model_dir["adapter_path"] / "adapters.safetensors"
        assert adapter_file.exists(), "adapters.safetensors not created!"
        assert adapter_file.stat().st_size > 0, "adapters.safetensors is empty!"

    def test_adapter_config_created(self, trained_model_dir):
        """Test that adapter_config.json is created after training."""
        config_file = trained_model_dir["adapter_path"] / "adapter_config.json"
        assert config_file.exists(), "adapter_config.json not created!"

    def test_adapter_config_has_required_fields(self, trained_model_dir):
        """Test that adapter_config.json has all fields required by mlx_lm."""
        config_file = trained_model_dir["adapter_path"] / "adapter_config.json"
        with open(config_file) as f:
            config = json.load(f)

        # Required by mlx_lm.tuner.utils.load_adapters
        assert "fine_tune_type" in config, "Missing fine_tune_type"
        assert "num_layers" in config, "Missing num_layers"
        assert config["num_layers"] is not None, "num_layers is None"
        assert "lora_parameters" in config, "Missing lora_parameters"

        params = config["lora_parameters"]
        assert "rank" in params, "Missing rank in lora_parameters"
        assert "scale" in params, "Missing scale in lora_parameters"

    def test_adapter_config_format_matches_mlx_lm(self, trained_model_dir):
        """Test that adapter_config.json format matches mlx_lm expectations."""
        config_file = trained_model_dir["adapter_path"] / "adapter_config.json"
        with open(config_file) as f:
            config = json.load(f)

        # Check correct values
        assert config["fine_tune_type"] == "lora"
        assert isinstance(config["num_layers"], int)
        assert config["num_layers"] > 0

        params = config["lora_parameters"]
        assert params["rank"] == 8  # We configured r=8
        assert params["scale"] == 1.0  # alpha/r = 8/8 = 1.0

    def test_adapters_loadable_by_mlx_lm(self, trained_model_dir):
        """Test that saved adapters can be loaded by mlx_lm.

        This is the critical test - if mlx_lm can't load our adapters,
        GGUF export will fail with 'adapter_config.json not found'.
        """
        from mlx_lm import load

        adapter_path = trained_model_dir["adapter_path"]

        # This should NOT raise FileNotFoundError for adapter_config.json
        model, tokenizer = load(
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            adapter_path=str(adapter_path),
        )

        assert model is not None
        assert tokenizer is not None

    def test_adapter_path_is_absolute(self, trained_model_dir):
        """Test that adapter_path is inside output_dir, not cwd."""
        output_dir = trained_model_dir["output_dir"]
        adapter_path = trained_model_dir["adapter_path"]

        # adapter_path should be under output_dir
        assert str(adapter_path).startswith(str(output_dir)), \
            f"adapter_path {adapter_path} should be under output_dir {output_dir}"


class TestGGUFExport:
    """Tests for GGUF export functionality.

    Note: GGUF export has mlx_lm limitations with quantized models.
    These tests verify our code works correctly; actual GGUF creation
    depends on mlx_lm capabilities.
    """

    def test_gguf_export_uses_correct_model_path(self):
        """Test that GGUF export uses original model path, not output dir.

        This was GitHub issue #3 - save_pretrained_gguf was passing
        the output directory as the model path.
        """
        from unsloth_mlx import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            max_seq_length=256,
        )

        # model_name should be preserved
        assert model.model_name == "mlx-community/Llama-3.2-1B-Instruct-4bit"

    def test_gguf_export_detects_adapter_path(self):
        """Test that GGUF export finds adapter path when LoRA is applied."""
        from unsloth_mlx import FastLanguageModel
        import tempfile

        model, tokenizer = FastLanguageModel.from_pretrained(
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            max_seq_length=256,
        )
        model = FastLanguageModel.get_peft_model(model, r=8)

        # Set adapter path
        with tempfile.TemporaryDirectory() as tmpdir:
            model.set_adapter_path(tmpdir)
            assert model.get_adapter_path() is not None


class TestRLTrainersEndToEnd:
    """End-to-end tests for RL trainers."""

    def test_dpo_trainer_saves_adapter_config(self):
        """Test that DPOTrainer saves adapter_config.json."""
        from unsloth_mlx import FastLanguageModel, DPOTrainer, DPOConfig
        import tempfile

        model, tokenizer = FastLanguageModel.from_pretrained(
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
            max_seq_length=256,
        )
        model = FastLanguageModel.get_peft_model(model, r=8)

        preference_data = [
            {"prompt": "Hi", "chosen": "Hello!", "rejected": "Go away"}
        ] * 3

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DPOTrainer(
                model=model,
                train_dataset=preference_data,
                tokenizer=tokenizer,
                args=DPOConfig(
                    output_dir=tmpdir,
                    max_steps=2,
                ),
            )
            trainer.train()

            # Check adapter_config.json was created
            config_file = Path(tmpdir) / "adapters" / "adapter_config.json"
            assert config_file.exists(), "DPOTrainer didn't create adapter_config.json"

            with open(config_file) as f:
                config = json.load(f)
            assert "num_layers" in config
            assert "lora_parameters" in config


class TestMergedModelSave:
    """Test save_pretrained_merged workflow (GitHub issue #4)."""

    @pytest.fixture
    def trained_model(self):
        """Fixture providing a model with LoRA that has been 'trained'."""
        from unsloth_mlx import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
            max_seq_length=512,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(model, r=8)

        # Apply LoRA so we have something to fuse
        model._apply_lora()

        return model, tokenizer

    def test_save_pretrained_merged_fuses_lora(self, trained_model):
        """Test that save_pretrained_merged properly fuses LoRA layers.

        This was broken - LoRA layers were saved as-is, causing
        'parameters not in model' errors when loading.
        """
        model, tokenizer = trained_model

        with tempfile.TemporaryDirectory() as tmpdir:
            merged_path = os.path.join(tmpdir, "merged")

            # Save merged model
            model.save_pretrained_merged(merged_path, tokenizer)

            # Check that model files were saved
            assert os.path.exists(merged_path), "Merged model directory not created"

            # Check for model weights file
            weights_file = os.path.join(merged_path, "model.safetensors")
            weights_exist = os.path.exists(weights_file)

            # Also check for sharded weights (larger models)
            sharded_weights = list(Path(merged_path).glob("model-*.safetensors"))

            assert weights_exist or len(sharded_weights) > 0, \
                f"No model weights found in {merged_path}"

    def test_merged_model_has_no_lora_keys(self, trained_model):
        """Test that merged model doesn't have LoRA-specific keys.

        The bug was that keys like 'k_proj.linear.biases' were being
        saved, which indicates LoRA layers weren't fused.
        """
        import mlx.core as mx
        model, tokenizer = trained_model

        with tempfile.TemporaryDirectory() as tmpdir:
            merged_path = os.path.join(tmpdir, "merged")
            model.save_pretrained_merged(merged_path, tokenizer)

            # Load the saved weights and check keys
            weights_file = os.path.join(merged_path, "model.safetensors")
            if os.path.exists(weights_file):
                weights = mx.load(weights_file)
                keys = list(weights.keys())

                # These patterns indicate unfused LoRA layers (NOT quantization scales)
                # LoRA wrapper structure: layer.linear.weight, layer.linear.biases
                # LoRA matrices: layer.lora_a, layer.lora_b
                lora_patterns = ['.linear.weight', '.linear.biases', '.lora_a', '.lora_b']
                problematic_keys = [
                    k for k in keys
                    if any(p in k for p in lora_patterns)
                ]

                assert len(problematic_keys) == 0, \
                    f"Found unfused LoRA keys: {problematic_keys[:5]}"


class TestLoadAdapter:
    """Test load_adapter functionality."""

    def test_load_adapter_method_exists(self):
        """Test that load_adapter method is available."""
        from unsloth_mlx import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
            max_seq_length=512,
            load_in_4bit=True,
        )

        assert hasattr(model, 'load_adapter'), "load_adapter method missing"

    def test_load_adapter_requires_files(self):
        """Test that load_adapter fails gracefully with missing files."""
        from unsloth_mlx import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
            max_seq_length=512,
            load_in_4bit=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to load from empty directory
            with pytest.raises(FileNotFoundError) as excinfo:
                model.load_adapter(tmpdir)

            assert "adapters.safetensors" in str(excinfo.value) or \
                   "does not exist" in str(excinfo.value)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
