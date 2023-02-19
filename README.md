# Convert Safetensor to Diffusers

This repo is for converting a CompVis checkpoint in safetensor format into files for [Diffusers](https://huggingface.co/docs/diffusers/index), edited from [diffuser space](https://huggingface.co/spaces/diffusers/convert-sd-ckpt)

## Install
```shell
$ pip install -r requirements.txt
```
## How to Use
```python
from utils import convert_full_checkpoint

safe_tensor_path = "/path/to/your-safe-tensor-model"
# replace None with the path to your vae.pt file if you want to use customized vae weights instead of those saved in safetensors
vae_pt_path = None

HF_MODEL_DIR = "/path/to/save/hf/model"

# noise scheduler you want to set as default
scheduler_type = "PNDM"  # K-LMS / DDIM / EulerAncestral / K-LMS

config_file = "./inference_config/v1-5-inference.yaml"

extract_ema = False

convert_full_checkpoint(
    safe_tensor_path,
    config_file,
    scheduler_type=scheduler_type,
    extract_ema=extract_ema,
    output_path=HF_MODEL_DIR,
    vae_pt_path=vae_pt_path,
)
```