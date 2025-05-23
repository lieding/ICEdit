import lightning as L
from diffusers.pipelines import FluxPipeline, FluxFillPipeline
import torch
from peft import LoraConfig, get_peft_model_state_dict
import os
import prodigyopt
import re

from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, encode_images_fill, prepare_text_input


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
            
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise
        
        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')
            
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        lora_layers = []
        try:
            if lora_config:
                self.transformer.add_adapter(LoraConfig(**lora_config))
            
            if lora_path:
                FluxFillPipeline.load_lora_weights(self.transformer, lora_path)
            
            lora_layers_iter = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
            # Convert iterator to list to allow multiple iterations if needed and for return
            lora_layers = list(lora_layers_iter)

            if lora_config and "target_modules" in lora_config:
                target_modules = lora_config["target_modules"]
                if not isinstance(target_modules, list): # Ensure target_modules is a list
                    target_modules = [target_modules]

                trainable_param_names = [
                    name for name, param in self.transformer.named_parameters() if param.requires_grad
                ]

                for name in trainable_param_names:
                    matched_a_target_module = False
                    # Extract the base module name before typical LoRA suffixes
                    # e.g., model.layers.0.self_attn.q_proj.lora_A.weight -> model.layers.0.self_attn.q_proj
                    base_name_parts = name.split(".lora_")
                    module_name_to_match = base_name_parts[0]

                    for pattern in target_modules:
                        if re.match(pattern, module_name_to_match):
                            matched_a_target_module = True
                            break 
                    
                    if not matched_a_target_module:
                        # Check if the trainable parameter is a bias term if 'bias' in lora_config
                        # and if it is one of 'all', 'lora_only'
                        bias_config = lora_config.get("bias", "none")
                        is_bias_param = "bias" in name.lower() # Simple check, PEFT might have more specific naming

                        if bias_config != "none" and is_bias_param:
                             # If bias terms are expected to be trainable, we assume this one is fine.
                             # A more precise check might involve cross-referencing with PEFT's internal logic
                             # for which bias parameters it makes trainable.
                             # For now, if bias is not 'none' and param name suggests bias, we accept it.
                            pass # Assume it's an intended trainable bias
                        else:
                            raise ValueError(
                                f"Trainable parameter {name} (base: {module_name_to_match}) "
                                f"does not match any target_modules pattern in lora_config ({target_modules}) "
                                f"and is not recognized as an expected trainable bias (bias config: {bias_config}). "
                                "Ensure LoRA configuration is correct."
                            )
                print("Successfully verified all trainable LoRA parameters match target_modules or expected bias terms.")

            return lora_layers
        except FileNotFoundError as e:
            print(f"Error: LoRA weights file not found at {lora_path}. Details: {e}")
            raise
        except IOError as e:
            print(f"Error: I/O error while loading LoRA weights from {lora_path}. Details: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during LoRA initialization. Details: {e}")
            raise

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )
            
            x_0, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)
                
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss
