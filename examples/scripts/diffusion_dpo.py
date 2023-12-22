# Copyright 2023 metric-space, The HuggingFace Team. All rights reserved.
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


from dataclasses import dataclass, field

import tyro

from trl import DefaultDiffusionDPOStableDiffusionPipeline, DiffusionDPOConfig, DiffusionDPOTrainer


@dataclass
class ScriptArguments:
    hf_user_access_token: str
    pretrained_model: str = "runwayml/stable-diffusion-v1-5"
    """the pretrained model to use"""
    pretrained_revision: str = "main"
    """the pretrained model revision to use"""
    hf_hub_model_id: str = "diffusion-dpo-finetuned-stable-diffusion"
    """HuggingFace repo to save model weights to"""

    print("Using manual beta and weight decay")
    diffusion_dpo_config: DiffusionDPOConfig = field(
        default_factory=lambda: DiffusionDPOConfig(
            tracker_project_name="stable_diffusion_training",
            beta_dpo=5000, train_adam_weight_decay=0,
            dataset_cache_dir="/export/share/datasets/vision_language/pick_a_pic_v2/",
            log_with="tensorboard",
            project_kwargs={
                "logging_dir": "./logs",
                "automatic_checkpoint_naming": True,
                "total_limit": 5,
                "project_dir": "./save",
            },
        )
    )


# def image_outputs_logger(image_data, global_step, accelerate_logger):
#     # For the sake of this example, we will only log the last batch of images
#     # and associated data
#     result = {}
#     images, prompts, _, rewards, _ = image_data[-1]

#     for i, image in enumerate(images):
#         prompt = prompts[i]
#         reward = rewards[i].item()
#         result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0)

#     accelerate_logger.log_images(
#         result,
#         step=global_step,
#     )


if __name__ == "__main__":
    args = tyro.cli(ScriptArguments)

    pipeline = DefaultDiffusionDPOStableDiffusionPipeline(
        args.pretrained_model, pretrained_model_revision=args.pretrained_revision, use_lora=False
    )

    trainer = DiffusionDPOTrainer(
        args.diffusion_dpo_config,
        pipeline,
        image_samples_hook=None,  # TODO
    )

    trainer.train()

    trainer.push_to_hub(args.hf_hub_model_id, token=args.hf_user_access_token)
