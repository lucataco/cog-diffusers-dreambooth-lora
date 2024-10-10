# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path, Secret
import os
import time
import subprocess
from zipfile import ZipFile

MODEL_NAME = "FLUX.1-dev"
MODEL_URL_DEV = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.system("accelerate config default --mixed_precision bf16")
        print("Loading Flux dev pipeline")
        if not os.path.exists("FLUX.1-dev"):
            download_weights(MODEL_URL_DEV, ".")

    # LoRA + DreamBooth
    def predict(
        self,
        input_images: Path = Input(description="A zip file containing the images that will be used for training.", default=None,),
        instance_prompt: str = Input(description="Instance prompt to trigger the image generation", default="a photo of TOK dog"),
        resolution: int = Input(description="The resolution for input images, all the images in the train/validation dataset will be resized to this", default=512, choices=[512,768,1024]),
        max_train_steps: int = Input(description="Total number of training steps to perform", default=100, ge=500, le=6000),
        rank: int = Input(description="The dimension of the LoRA", default=4, ge=4, le=64),
        train_batch_size: int = Input(description="Batch size for the training dataloader", default=1, ge=1, le=8),
        gradient_accumulation_steps: int = Input(description="Number of updates steps to accumulate before performing a backward/update pass", default=1, ge=1, le=8),
        optimizer: str = Input(description="The optimizer type to use", default="AdamW", choices=["AdamW", "prodigy"]),
        learning_rate: float = Input(description="Initial learning rate to use (1.0 for Prodigy)", default=0.0001, ge=0.0001, le=1),
        lr_scheduler: str = Input(description="'The scheduler type to use", default="constant", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]),
        checkpointing_steps: int = Input(description="Save a checkpoint of the training state every X updates", default=None, ge=100, le=6000),
        seed: int = Input(description="Seed for reproducibility", default=None),
        backend: str = Input(description="Dynamo Backend", default="no", choices=["no", "eager", "aot_eager", "inductor", "nvfuser", "aot_nvfuser", "aot_cudagraphs", "ofi", "fx2trt", "onnxrt", "ipex"]),
        hf_token: Secret = Input(description="Huggingface token (optional) with write access to upload to Hugging Face", default=None),
        hub_model_id: str = Input(description="Huggingface model location for upload. Requires a HF token with write permissions. Ex: lucataco/flux-qsd", default=None),
        wandb_api_key: Secret = Input(description="Weights and Biases API key, if you'd like to log training progress to W&B.", default=None)
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        # Cleanup
        input_dir = "input_images"
        output_base = "/tmp/train"
        output_logs = "/tmp/logs"
        output_dir = output_base + "/output/flux_train_replicate"
        os.system(f"rm -rf {input_dir}")
        os.system(f"rm -rf {output_base}")
        os.system(f"rm -rf {output_logs}")

        # Check input images zip file
        if not input_images.name.endswith(".zip"):
            raise ValueError("input_images must be a zip file")
        # Extract images from zip file
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

        # Dont save checkpoints by default
        if checkpointing_steps is None:
            checkpointing_steps = max_train_steps+1

        # Check optimizer
        if optimizer == "prodigy":
            learning_rate = 1.0
        # Trainer params
        run_params = [
            "accelerate",
            "launch",
            "--dynamo_backend", backend,
            "train_dreambooth_lora_flux.py",
            "--pretrained_model_name_or_path", MODEL_NAME,
            "--instance_data_dir", str(input_dir),
            "--rank", str(rank),
            "--output_dir", output_dir,
            "--mixed_precision", "bf16",
            "--instance_prompt", instance_prompt,
            "--resolution", str(resolution),
            "--train_batch_size", str(train_batch_size),
            "--gradient_accumulation_steps", str(gradient_accumulation_steps),
            "--optimizer=prodigy",
            "--learning_rate", str(learning_rate),
            "--lr_scheduler", lr_scheduler,
            "--lr_warmup_steps", "0",
            "--max_train_steps", str(max_train_steps),
            "--checkpointing_steps", str(checkpointing_steps),
            "--seed", str(seed),
            "--logging_dir", output_logs
        ]
        # Check for
        # if optimizer == "AdamW":
        #     run_params.extend(["--use_8bit_adam"])

        # Check to upload to HF
        if hf_token is not None and hub_model_id is not None:
            token = hf_token.get_secret_value()
            os.system(f"huggingface-cli login --token {token}")
            run_params.extend(["--push_to_hub"])
            run_params.extend(["--hub_token", token])
            run_params.extend(["--hub_model_id", hub_model_id])

        # Check to log training run to Wandb
        if wandb_api_key:
            api_key = wandb_api_key.get_secret_value()
            os.environ["WANDB_API_KEY"] = api_key
            run_params.extend(['--report_to', 'wandb'])
        
        # Run the trainer
        print(f"Using params: {run_params}")
        subprocess.run(run_params, check=True, close_fds=False)

        # rename safetensors to lora_weights.safetensors
        if os.path.exists(f"{output_dir}/pytorch_lora_weights.safetensors"):
            os.system(f"mv {output_dir}/pytorch_lora_weights.safetensors {output_dir}/lora.safetensors")
        
        # Copy Lora license if uploading to HF
        if hub_model_id is not None:
            os.system(f"cp lora-license.md {output_dir}/README.md")

        # Create uploadable tar
        output_path = "/tmp/trained_model.tar"
        os.system(f"tar -cvf {output_path} -C {output_base} .")
        return Path(output_path)
