# -*- coding:utf-8 -*-
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# edited by Norm Xu
""" Conversion script for the Stable Diffusion checkpoints. """

from utils import convert_full_checkpoint

if __name__ == '__main__':
    safe_tensor_path = "/path/to/safe-tensor"
    vae_pt_path = "/path/to/vae"
    HF_MODEL_DIR = "/path/to/save/hf/model"
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
        with_control_net=False
    )
