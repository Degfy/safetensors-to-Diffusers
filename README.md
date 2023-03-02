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

# replace None with the path to your vae.pt file you like
vae_pt_path = None

HF_MODEL_DIR = "/path/to/save/hf/model"

# noise scheduler you want to set as default
scheduler_type = "PNDM"  # K-LMS / DDIM / EulerAncestral / K-LMS

# use the corresponding sd config file that your model is fine-tuned based on
config_file = "./inference_config/v1-5-inference.yaml"

extract_ema = False

convert_full_checkpoint(
    safe_tensor_path,
    config_file,
    scheduler_type=scheduler_type,
    extract_ema=extract_ema,
    output_path=HF_MODEL_DIR,
    vae_pt_path=vae_pt_path,
    with_control_net=False
)
```

## Notes
If your checkpoint contains a ControlNet, such as [these checkpoints](https://huggingface.co/lllyasviel/ControlNet/tree/main/models), set ``with_control_net=True`` to seperate ControlNet parameters from the checkpoint.  ControlNet can be regarded as a variant of UNet2DConditionModel, therefore, most codes for converting UNet2DConditionModel can be reused.  
