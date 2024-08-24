# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path, Secret
import os
import subprocess
from zipfile import ZipFile

# Set your HF_TOKEN
HF_TOKEN = os.environ["HF_TOKEN"]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.system("accelerate config default --mixed_precision bf16")
        print("Loading Flux dev weights")
        os.system(f"huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir FLUX.1-dev --token {HF_TOKEN}")

    def predict(
        self,
        input_images: Path = Input(
            description="A zip file containing the images that will be used for training.",
            default=None,
        ),
        instance_prompt: str = Input(description="Instance prompt to trigger the image generation", default="a photo of TOK dog"),
        resolution: int = Input(description="The resolution for input images, all the images in the train/validation dataset will be resized to this", default=768, ge=128, le=1024),
        train_batch_size: int = Input(description="Batch size for the training dataloader", default=1, ge=1, le=8),
        gradient_accumulation_steps: int = Input(description="Number of updates steps to accumulate before performing a backward/update pass", default=1, ge=1, le=8),
        learning_rate: float = Input(description="Initial learning rate to use", default=0.0001, ge=0.0001, le=0.01),
        lr_scheduler: str = Input(description="'The scheduler type to use", default="constant", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]),
        max_train_steps: int = Input(description="Total number of training steps to perform", default=500, ge=10, le=6000),
        checkpointing_steps: int = Input(description="Save a checkpoint of the training state every X updates", default=None, ge=100, le=6000),
        seed: int = Input(description="Seed for reproducibility", default=None),
        hf_token: Secret = Input(description="Huggingface token (optional) with write access to upload to HF", default=None),
        hub_model_id: str = Input(description="Huggingface model location for upload. Requires HF token. Ex: lucataco/dreambooth-lora", default=None)
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        model_name="FLUX.1-dev"

        # Cleanup
        input_dir = "input_images"
        output_base = "/tmp/train"
        output_dir = output_base + "/output/flux_train_replicate"
        os.system(f"rm -rf {input_dir}")
        os.system(f"rm -rf {output_base}")

        if not input_images.name.endswith(".zip"):
            raise ValueError("input_images must be a zip file")
        
        input_dir = Path(input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        image_count = 0
        with ZipFile(input_images, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                if not file_info.filename.startswith(
                    "__MACOSX/"
                ) and not file_info.filename.startswith("._"):
                    zip_ref.extract(file_info, input_dir)
                    image_count += 1
        print(f"Extracted {image_count} files from zip to {input_dir}")

        if checkpointing_steps is None:
            checkpointing_steps = max_train_steps+1

        run_params = [
            "accelerate", "launch", "train_dreambooth_lora_flux.py",
            "--pretrained_model_name_or_path", model_name,
            "--instance_data_dir", str(input_dir),
            "--output_dir", output_dir,
            "--mixed_precision", "bf16",
            "--instance_prompt", instance_prompt,
            "--resolution", str(resolution),
            "--train_batch_size", str(train_batch_size),
            "--gradient_accumulation_steps", str(gradient_accumulation_steps),
            "--learning_rate", str(learning_rate),
            "--lr_scheduler", lr_scheduler,
            "--lr_warmup_steps", "0",
            "--max_train_steps", str(max_train_steps),
            "--checkpointing_steps", str(checkpointing_steps),
            "--seed", str(seed),
            "--use_8bit_adam",
            "--logging_dir", "/tmp/logs"
        ]
        if hf_token is not None and hub_model_id is not None:
            token = hf_token.get_secret_value()
            os.system(f"huggingface-cli login --token {token}")
            run_params.extend(["--push_to_hub"])
            run_params.extend(["--hub_token", token])
            run_params.extend(["--hub_model_id", hub_model_id])
        
        print(f"Using params: {run_params}")
        subprocess.check_call(run_params)

        os.system(f"mv {output_dir}/pytorch_lora_weights.safetensors {output_dir}/lora.safetensors")
        
        if hub_model_id is not None:
            os.system(f"cp lora-license.md {output_dir}/README.md")

        output_path = "/tmp/trained_model.tar"
        os.system(f"tar -cvf {output_path} -C {output_base} .")
        return Path(output_path)
