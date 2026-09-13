import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import pytorch_lightning as pl
import clip.modules.vision_transformer_prompts as vit
import math
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from clip.modules import clip_utils, heads, objectives, clip
from clip.modules.counterfactual_proxy import CounterfactualProxyGame
from clip.modules.reliability_learning import (
    PromptAdapter,
    ReliabilityPredictor,
    UtilityGate,
    availability_from_missing_type,
    endpoint_gate_target,
    regret_weighted_binary_gate_loss,
    reliability_targets_from_delta,
    soft_gate_target,
    validate_gate_candidates,
    validate_gate_supervision_mode,
    validate_reliability_importance_mode,
)
import copy


def training_mode_from_config(config):
    reliability_enabled = bool(config.get("reliability_enabled", False))
    counterfactual_enabled = bool(config.get("counterfactual_enabled", False))
    if reliability_enabled and counterfactual_enabled:
        raise ValueError(
            "reliability_enabled and counterfactual_enabled cannot be enabled together"
        )
    if reliability_enabled:
        active_tasks = sorted(
            name
            for name, value in config.get("loss_names", {}).items()
            if float(value) >= 1.0
        )
        if active_tasks != ["mmimdb"]:
            raise ValueError(
                "Reliability training currently supports only the MM-IMDb task; "
                "active tasks are {}".format(active_tasks)
            )
        return "reliability"
    if counterfactual_enabled:
        return "counterfactual"
    return "dcp"

def load_clip_to_cpu(backbone_name, prompt_length, prompt_depth, cache_root=None):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root=cache_root) if cache_root else clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")#.eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = vit.build_model(state_dict or model.state_dict(), prompt_length, prompt_depth)

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.prompt_length = clip_model.prompt_length

    def forward(self, tokenized_texts, all_prompts_text, missing_type):
        x = self.token_embedding(tokenized_texts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, all_prompts_text, 0, missing_type]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0][self.prompt_length:]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_texts.argmax(dim=-1)] @ self.text_projection

        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiModalPromptLearner(nn.Module):
    def __init__(
        self,
        prompt_length,
        prompt_depth,
        clip_model,
        reliability_enabled=False,
        reliability_prompt_hidden_dim=64,
        adapter_scale=1.0,
        reliability_importance_mode="absolute",
    ):
        super().__init__()
        dtype = clip_model.dtype
        prompt_length_half = prompt_length//3 # use half length for generating static prompts, and the other for generating dynamic prompts
        # Default is 1, which is compound shallow prompting
        self.prompt_depth = prompt_depth  # max=12, but will create 11 such shared prompts
        self.visual_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.visual_prompt_missing = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.text_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.text_prompt_missing = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_image = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_text = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        # Also make corresponding projection layers, for each prompt
        embed_dim_text = 512
        embed_dim_image = 768
        embed_dim = embed_dim_text + embed_dim_image
        r = 16
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_text),
                )
        self.compound_prompt_projections_text = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_text = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_image),
                )
        self.compound_prompt_projections_image = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_image = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        self.common_prompt_projection_image = nn.Sequential(
                nn.Linear(embed_dim_text, embed_dim_text//r),
                nn.GELU(),
                nn.Linear(embed_dim_text//r, embed_dim_image),
                )
        self.common_prompt_projection_text = nn.Sequential(
                nn.Linear(embed_dim_text, embed_dim_text//r),
                nn.GELU(),
                nn.Linear(embed_dim_text//r, embed_dim_text),
                )
        self.reliability_enabled = bool(reliability_enabled)
        if self.reliability_enabled:
            self.prompt_adapter = PromptAdapter(
                prompt_depth=self.prompt_depth,
                image_prompt_dim=self.visual_prompt_complete.shape[-1],
                text_prompt_dim=self.text_prompt_complete.shape[-1],
                hidden_dim=reliability_prompt_hidden_dim,
                scale=adapter_scale,
                use_availability=(reliability_importance_mode == "relative"),
            )

    def forward(
        self,
        missing_type,
        reliability=None,
        availability_mask=None,
        utility_gate=None,
        return_adapter_diagnostics=False,
    ):

        # Before returning, need to transform
        # prompts to 768 for the visual side
        all_prompts_image = [ [] for _ in range(self.prompt_depth)]   # Prompts of prompt_depth layers
        all_prompts_text = [ [] for _ in range(self.prompt_depth)]   # Prompts of prompt_depth layers
        for i in range(len(missing_type)):
            # set initial prompts for each modality
            if missing_type[i]==0:  # modality complete
                initial_prompt_image = self.visual_prompt_complete
                initial_prompt_text = self.text_prompt_complete
                common_prompt = self.common_prompt_complete
            elif missing_type[i]==1:  # missing text 
                initial_prompt_image = self.visual_prompt_complete
                initial_prompt_text = self.text_prompt_missing
                common_prompt = self.common_prompt_image
            elif missing_type[i]==2:  # missing image 
                initial_prompt_image = self.visual_prompt_missing
                initial_prompt_text = self.text_prompt_complete
                common_prompt = self.common_prompt_text
            # generate the prompts of the first layer
            all_prompts_image[0].append(self.compound_prompt_projections_image[0](self.layernorm_image[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))
            all_prompts_text[0].append(self.compound_prompt_projections_text[0](self.layernorm_text[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))
            # generate the prompts of the rest layers
            for index in range(1, self.prompt_depth):
                all_prompts_image[index].append(
                    self.compound_prompt_projections_image[index](self.layernorm_image[index](torch.cat([all_prompts_image[index-1][-1], all_prompts_text[index-1][-1]], -1))))
                all_prompts_text[index].append(
                    self.compound_prompt_projections_text[index](self.layernorm_text[index](torch.cat([all_prompts_image[index-1][-1], all_prompts_text[index-1][-1]], -1))))
            all_prompts_image[0][i] = torch.cat([
                    all_prompts_image[0][i], 
                    self.common_prompt_projection_image(common_prompt)]
                    ,0)
            all_prompts_text[0][i] = torch.cat([
                    all_prompts_text[0][i], 
                    self.common_prompt_projection_text(common_prompt)]
                    ,0)
        # generate the prompts in each layer as a tensor [B, L, C]
        all_prompts_image = [torch.stack(prompts) for prompts in all_prompts_image]
        all_prompts_text = [torch.stack(prompts) for prompts in all_prompts_text]
        if reliability is not None:
            if not self.reliability_enabled:
                raise RuntimeError("Reliability prompt adaptation is not enabled")
            if availability_mask is None or utility_gate is None:
                raise ValueError(
                    "availability_mask and utility_gate are required with reliability"
                )
            adapted = self.prompt_adapter(
                all_prompts_image, all_prompts_text, reliability,
                availability_mask, utility_gate,
                return_diagnostics=return_adapter_diagnostics,
            )
            if return_adapter_diagnostics:
                all_prompts_image, all_prompts_text, adapter_diagnostics = adapted
                return all_prompts_image, all_prompts_text, adapter_diagnostics
            all_prompts_image, all_prompts_text = adapted
        if return_adapter_diagnostics:
            return all_prompts_image, all_prompts_text, []
        return all_prompts_image, all_prompts_text

class CustomCLIP(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model, config=None):
        super().__init__()
        config = config or {}
        self.counterfactual_enabled = config.get("counterfactual_enabled", False)
        self.reliability_enabled = bool(config.get("reliability_enabled", False))
        self.reliability_importance_mode = validate_reliability_importance_mode(
            config.get("reliability_importance_mode", "absolute")
        )
        self.gate_candidates = validate_gate_candidates(
            config.get("gate_candidates", [0.0, 0.25, 0.5, 0.75, 1.0])
        )
        if self.counterfactual_enabled and self.reliability_enabled:
            raise ValueError(
                "reliability_enabled and counterfactual_enabled are independent "
                "experimental modes and cannot be enabled together"
            )

        self.prompt_learner = MultiModalPromptLearner(
            prompt_length,
            prompt_depth,
            clip_model,
            reliability_enabled=self.reliability_enabled,
            reliability_prompt_hidden_dim=config.get(
                "reliability_prompt_hidden_dim", 64
            ),
            adapter_scale=config.get("adapter_scale", 1.0),
            reliability_importance_mode=self.reliability_importance_mode,
        )
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.image_feature_dim = int(self.image_encoder.output_dim)
        self.text_feature_dim = int(self.text_encoder.text_projection.shape[1])

        if self.reliability_enabled:
            self.reliability_predictor = ReliabilityPredictor(
                image_feature_dim=self.image_feature_dim,
                text_feature_dim=self.text_feature_dim,
                hidden_dim=config.get("reliability_hidden_dim", 256),
                importance_mode=self.reliability_importance_mode,
            )
            self.utility_gate = UtilityGate(
                context_dim=self.image_feature_dim + self.text_feature_dim,
                projector_dim=64,
                hidden_dim=64,
            )

        if self.counterfactual_enabled:
            self.counterfactual_proxy = CounterfactualProxyGame(
                feature_dim=512,
                hidden_dim=config.get("counterfactual_hidden_dim", 512),
                identity_dim=config.get("counterfactual_identity_dim", 32),
                challenge_radius=config.get("counterfactual_challenge_radius", 1.0),
                adversary_strength=config.get("counterfactual_adversary_strength", 1.0),
            )

    @staticmethod
    def _mask_features(image_features, text_features, availability_mask):
        availability = availability_mask.to(
            device=image_features.device, dtype=image_features.dtype
        )
        image_features = image_features * availability[:, 0:1]
        text_features = text_features * availability[:, 1:2].to(text_features.dtype)
        return image_features, text_features

    def _tokenize_text(self, text, device):
        return torch.stack(
            [clip.tokenize(tx, context_length=77, truncate=True) for tx in text[0]],
            0,
        ).to(device).squeeze(1)

    def _encode_with_tokenized_text(
        self,
        image,
        tokenized_texts,
        missing_type,
        reliability=None,
        availability_mask=None,
        utility_gate=None,
        return_adapter_diagnostics=False,
    ):
        prompt_output = self.prompt_learner(
            missing_type,
            reliability=reliability,
            availability_mask=availability_mask,
            utility_gate=utility_gate,
            return_adapter_diagnostics=return_adapter_diagnostics,
        )
        if return_adapter_diagnostics:
            all_prompts_image, all_prompts_text, adapter_diagnostics = prompt_output
        else:
            all_prompts_image, all_prompts_text = prompt_output
        text_features = self.text_encoder(tokenized_texts, all_prompts_text, missing_type)
        image_features = self.image_encoder(image.type(self.dtype), all_prompts_image, missing_type)
        if return_adapter_diagnostics:
            return image_features, text_features, adapter_diagnostics
        return image_features, text_features

    def _encode_from_prompts(
        self, image, tokenized_texts, missing_type, image_prompts, text_prompts
    ):
        text_features = self.text_encoder(tokenized_texts, text_prompts, missing_type)
        image_features = self.image_encoder(
            image.type(self.dtype), image_prompts, missing_type
        )
        return image_features, text_features

    def _encode_base(self, image, text, missing_type):
        tokenized_texts = self._tokenize_text(text, image.device)
        return self._encode_with_tokenized_text(
            image, tokenized_texts, missing_type
        )

    def base_view(self, image, text, missing_type):
        """Run the frozen Original DCP path without Reliability components."""

        availability = availability_from_missing_type(
            missing_type, image.device, dtype=image.dtype
        )
        image_features, text_features = self._encode_base(image, text, missing_type)
        masked_image, masked_text = self._mask_features(
            image_features, text_features, availability
        )
        return {
            "cls_feats": torch.cat([masked_image, masked_text], dim=-1),
            "image_features": image_features,
            "text_features": text_features,
            "availability_mask": availability,
        }

    def prepare_reliability_state(
        self, image, text, missing_type, return_adapter_diagnostics=False
    ):
        if not self.reliability_enabled:
            raise RuntimeError("Reliability learning is not enabled")
        availability = availability_from_missing_type(
            missing_type, image.device, dtype=image.dtype
        )
        tokenized_texts = self._tokenize_text(text, image.device)
        with torch.no_grad():
            base_image, base_text = self._encode_with_tokenized_text(
                image, tokenized_texts, missing_type
            )
        base_image = base_image.detach()
        base_text = base_text.detach()
        reliability_logits, reliability = self.reliability_predictor(
            base_image, base_text, availability
        )
        masked_image, masked_text = self._mask_features(
            base_image, base_text, availability
        )
        base_context = torch.cat([masked_image, masked_text], dim=-1).detach()
        image_prompts, text_prompts = self.prompt_learner(missing_type)
        image_offsets, text_offsets, adapter_diagnostics = (
            self.prompt_learner.prompt_adapter.compute_offsets(
                image_prompts,
                text_prompts,
                reliability.detach(),
                availability,
                return_diagnostics=return_adapter_diagnostics,
            )
        )
        return {
            "image": image,
            "missing_type": missing_type,
            "tokenized_texts": tokenized_texts,
            "availability_mask": availability,
            "base_image_features": base_image,
            "base_text_features": base_text,
            "base_cls_feats": base_context,
            "reliability_logits": reliability_logits,
            "reliability": reliability,
            "image_prompts": image_prompts,
            "text_prompts": text_prompts,
            "image_offsets": image_offsets,
            "text_offsets": text_offsets,
            "adapter_diagnostics": adapter_diagnostics,
        }

    def reliability_view(self, image, text, missing_type):
        """Predict Reliability from a frozen Original-DCP representation."""

        with torch.no_grad():
            base = self.base_view(image, text, missing_type)
        logits, reliability = self.reliability_predictor(
            base["image_features"].detach(),
            base["text_features"].detach(),
            base["availability_mask"],
        )
        base.update({"reliability_logits": logits, "reliability": reliability})
        return base

    def encode_reliability_state(self, state, gate):
        base_features = state["base_cls_feats"]
        gate_values = gate.to(base_features).reshape(-1)
        batch_size = base_features.shape[0]
        if gate_values.numel() == 1:
            zero_gate_rows = gate_values.eq(0).expand(batch_size)
        elif gate_values.numel() == batch_size:
            zero_gate_rows = gate_values.eq(0)
        else:
            raise ValueError(
                "gate must contain one value or one value per batch sample"
            )
        if bool(zero_gate_rows.all()):
            return base_features

        image_prompts = PromptAdapter._apply_gate(
            state["image_prompts"], state["image_offsets"], gate
        )
        text_prompts = PromptAdapter._apply_gate(
            state["text_prompts"], state["text_offsets"], gate
        )
        image_features, text_features = self._encode_from_prompts(
            state["image"],
            state["tokenized_texts"],
            state["missing_type"],
            image_prompts,
            text_prompts,
        )
        image_features, text_features = self._mask_features(
            image_features, text_features, state["availability_mask"]
        )
        adapted_features = torch.cat([image_features, text_features], dim=-1)
        if bool(zero_gate_rows.any()):
            adapted_features = torch.where(
                zero_gate_rows.unsqueeze(-1), base_features, adapted_features
            )
        return adapted_features

    def forward(
        self,
        image,
        text,
        missing_type,
        training_phase="gate",
        gate_override=None,
        return_adapter_diagnostics=False,
    ):
        if self.reliability_enabled:
            state = self.prepare_reliability_state(
                image,
                text,
                missing_type,
                return_adapter_diagnostics=return_adapter_diagnostics,
            )
            utility_gate_logits = self.utility_gate.logits(
                state["reliability"],
                state["availability_mask"],
                state["base_cls_feats"],
            )
            predicted_gate = torch.sigmoid(utility_gate_logits)
            if gate_override is not None:
                utility_gate = gate_override.to(predicted_gate).view(-1, 1)
            elif training_phase == "adapter":
                utility_gate = torch.ones_like(predicted_gate)
            else:
                utility_gate = predicted_gate
            cls_feats = self.encode_reliability_state(state, utility_gate)
            return {
                "cls_feats": cls_feats,
                "base_cls_feats": state["base_cls_feats"],
                "base_image_features": state["base_image_features"],
                "base_text_features": state["base_text_features"],
                "reliability_logits": state["reliability_logits"],
                "reliability": state["reliability"],
                "availability_mask": state["availability_mask"],
                "utility_gate": utility_gate,
                "utility_gate_logits": utility_gate_logits,
                "predicted_utility_gate": predicted_gate,
                "adapter_diagnostics": state["adapter_diagnostics"],
                "reliability_state": state,
            }

        tokenized_texts = self._tokenize_text(text, image.device)
        image_features, text_features = self._encode_with_tokenized_text(
            image, tokenized_texts, missing_type
        )
        if self.counterfactual_enabled:
            return self.counterfactual_proxy(image_features, text_features, missing_type)
        return {"cls_feats": torch.cat([image_features, text_features], -1)}

class CLIPransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.training_mode = training_mode_from_config(config)
        self.gate_supervision_mode = validate_gate_supervision_mode(
            config.get("gate_supervision_mode", "soft_scalar")
        )
        self.automatic_optimization = self.training_mode != "reliability"
        self.save_hyperparameters()

        clip_model = load_clip_to_cpu(
            config['vit'], config['prompt_length'], config['prompt_depth'],
            config.get('clip_cache_root') or None,
        )

        print("Building custom CLIP")
        self.model = CustomCLIP(config['prompt_length'], config['prompt_depth'], clip_model, config)
        hidden_size = self.model.image_feature_dim + self.model.text_feature_dim
        self._gate_epoch_predictions = []
        self._gate_epoch_targets = []
        self._gate_epoch_hard_targets = []
        self._epoch_metric_values = {}

        # ===================== Downstream ===================== #
        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Linear(hidden_size, cls_num)
            self.hatememes_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Linear(hidden_size, cls_num)
            self.food101_classifier.apply(objectives.init_weights)               
            
        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Linear(hidden_size, cls_num)
            self.mmimdb_classifier.apply(objectives.init_weights)  

        clip_utils.set_metrics(self)
        self.current_tasks = list()
        if self.hparams.config.get("test_only", False):
            final_path = self.hparams.config.get("load_path", "")
            if not final_path:
                raise ValueError("test_only requires load_path for the final model")
            state_dict = torch.load(final_path, map_location="cpu")["state_dict"]
            self.load_state_dict(state_dict, strict=True)
        elif self.training_mode == "reliability":
            self._load_original_dcp_base()
            self._set_training_phase("adapter")
        else:
            self._load_standard_model_checkpoint()
            self._set_standard_trainable_parameters()
        self.records = {}

    def _load_standard_model_checkpoint(self):
        path = self.hparams.config.get("load_path", "")
        if not path:
            return
        if not os.path.isfile(path):
            raise FileNotFoundError("load_path checkpoint does not exist: {}".format(path))
        checkpoint = torch.load(path, map_location="cpu")
        source = checkpoint.get("state_dict", checkpoint)
        if not isinstance(source, dict):
            raise RuntimeError("load_path does not contain a state dictionary")
        if any(key.startswith("model.") for key in source):
            source = {
                key[len("model."):]: value
                for key, value in source.items()
                if key.startswith("model.")
            }
        incompatible = self.model.load_state_dict(source, strict=False)
        allowed_missing_prefixes = (
            ("counterfactual_proxy.",)
            if self.training_mode == "counterfactual"
            else ()
        )
        missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Standard model checkpoint mismatch: missing={} unexpected={}".format(
                    missing, sorted(incompatible.unexpected_keys)
                )
            )
        if self.hparams.config.get("finetune_first", False):
            print("use pre-finetune model")

    def _set_standard_trainable_parameters(self):
        for name, parameter in self.model.named_parameters():
            trainable = (
                "prompt_learner" in name
                or "prompt" in name
                or "ln_final" in name
                or "ln_post" in name
                or name.split(".")[-1] == "proj"
                or (
                    self.training_mode == "counterfactual"
                    and "counterfactual_proxy" in name
                )
            )
            parameter.requires_grad_(trainable)

    @staticmethod
    def _new_module_key(name):
        return name.startswith(
            (
                "model.reliability_predictor.",
                "model.prompt_learner.prompt_adapter.",
                "model.utility_gate.",
            )
        )

    def _load_original_dcp_base(self):
        path = self.hparams.config.get("original_dcp_path", "")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError("original_dcp_path does not exist: {}".format(path))
        source = torch.load(path, map_location="cpu")["state_dict"]
        current = self.state_dict()
        expected = {key for key in current if not self._new_module_key(key)}
        provided = set(source)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        shape_mismatches = sorted(
            key for key in expected & provided
            if tuple(current[key].shape) != tuple(source[key].shape)
        )
        if missing or unexpected or shape_mismatches:
            raise RuntimeError(
                "Original DCP base mismatch: missing={} unexpected={} shapes={}".format(
                    missing, unexpected, shape_mismatches
                )
            )
        current.update(source)
        self.load_state_dict(current, strict=True)

    def _set_training_phase(self, phase):
        if self.training_mode != "reliability":
            raise RuntimeError("training phases are only used in Reliability mode")
        if phase not in ("adapter", "gate"):
            raise ValueError("training phase must be adapter or gate")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if phase == "adapter":
            for parameter in self.model.reliability_predictor.parameters():
                parameter.requires_grad_(True)
            for parameter in self.model.prompt_learner.prompt_adapter.parameters():
                parameter.requires_grad_(True)
        else:
            for parameter in self.model.utility_gate.parameters():
                parameter.requires_grad_(True)
        self._training_phase = phase

    def _phase_for_epoch(self):
        boundary = int(self.hparams.config.get("adapter_train_epochs", 10))
        return "adapter" if int(self.current_epoch) < boundary else "gate"

    def on_train_epoch_start(self):
        if self.training_mode != "reliability":
            return
        phase = self._phase_for_epoch()
        self._set_training_phase(phase)
        self._epoch_metric_values = {}
        self.log("training_phase", 0.0 if phase == "adapter" else 1.0)
        print("TRAINING_PHASE={}".format(phase.upper()))
        if phase == "gate":
            print("GATE_SUPERVISION={}".format(self.gate_supervision_mode.upper()))

    def _buffer_epoch_metric(self, name, value):
        self._epoch_metric_values.setdefault(name, []).append(
            value.detach().float().mean().cpu()
        )

    def infer(self, batch, gate_override=None):
        text = batch["text"]
        img = batch["image"][0]  # extract the first view (total 1)
        if self.hparams.config["test_only"]:
            self.model.eval()
            if self.hparams.config["loss_names"]["hatememes"] > 0:
                self.hatememes_classifier.eval()
            
            if self.hparams.config["loss_names"]["food101"] > 0:
                self.food101_classifier.eval()            
            
            if self.hparams.config["loss_names"]["mmimdb"] > 0:
                self.mmimdb_classifier.eval()
        phase = "gate"
        if self.training_mode == "reliability":
            phase = "gate" if not self.training and self.hparams.config.get(
                "test_only", False
            ) else self._phase_for_epoch()
        model_output = self.model(
            img,
            text,
            batch["missing_type"],
            training_phase=phase,
            gate_override=gate_override,
        )
        if self.hparams.config.get("reliability_enabled", False):
            return model_output
        if self.hparams.config.get("counterfactual_enabled", False):
            return model_output

        both_feats = model_output["cls_feats"]
        feature_dim = both_feats.shape[1]//2
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                pass       
            elif batch["missing_type"][idx] == 1:  # missing text
                both_feats[idx, feature_dim:].zero_()
            elif batch["missing_type"][idx] == 2:
                both_feats[idx, :feature_dim].zero_()
            
        ret = {
            "cls_feats": both_feats,
        }

        return ret

    @staticmethod
    def _labels(batch, device):
        return torch.tensor(batch["label"], device=device).float()

    @staticmethod
    def _sample_loss(logits, labels):
        return F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        ).mean(dim=-1)

    @staticmethod
    def _select_text(text, indices):
        return ([text[0][int(index)] for index in indices],)

    def _compute_reliability_loss(self, batch):
        image = batch["image"][0]
        text = batch["text"]
        missing = torch.as_tensor(batch["missing_type"], device=image.device).long()
        labels = self._labels(batch, image.device)
        predicted = self.model.reliability_view(image, text, batch["missing_type"])
        availability = predicted["availability_mask"]
        with torch.no_grad():
            current_logits = self.mmimdb_classifier(predicted["cls_feats"])
            current_loss = self._sample_loss(current_logits, labels)
            null_logits = self.mmimdb_classifier(
                torch.zeros_like(predicted["cls_feats"])
            )
            null_loss = self._sample_loss(null_logits, labels)
            loss_without_image = null_loss.clone()
            loss_without_text = null_loss.clone()
            complete = missing.eq(0).nonzero(as_tuple=False).flatten()
            if complete.numel():
                indices = complete.detach().cpu().tolist()
                selected_image = image.index_select(0, complete)
                selected_text = self._select_text(text, indices)
                selected_labels = labels.index_select(0, complete)
                text_only = self.model.base_view(
                    torch.ones_like(selected_image),
                    selected_text,
                    [2] * len(indices),
                )
                image_only = self.model.base_view(
                    selected_image,
                    ([""] * len(indices),),
                    [1] * len(indices),
                )
                loss_without_image.index_copy_(
                    0,
                    complete,
                    self._sample_loss(
                        self.mmimdb_classifier(text_only["cls_feats"]),
                        selected_labels,
                    ),
                )
                loss_without_text.index_copy_(
                    0,
                    complete,
                    self._sample_loss(
                        self.mmimdb_classifier(image_only["cls_feats"]),
                        selected_labels,
                    ),
                )
            task_temperatures = self.hparams.config.get(
                "reliability_temperature_by_task", {}
            )
            temperature = task_temperatures.get("mmimdb")
            if temperature is None:
                temperature = self.hparams.config.get(
                    "reliability_temperature", 0.1
                )
            temperature = float(temperature)
            delta = torch.stack(
                [loss_without_image - current_loss, loss_without_text - current_loss],
                dim=-1,
            )
            target, supervision = reliability_targets_from_delta(
                delta,
                availability,
                temperature,
                importance_mode=self.model.reliability_importance_mode,
            )
        element_loss = F.binary_cross_entropy_with_logits(
            predicted["reliability_logits"], target, reduction="none"
        )
        loss = (element_loss * supervision).sum() / supervision.sum().clamp_min(1.0)
        self._buffer_epoch_metric("reliability_loss", loss)
        self._buffer_epoch_metric("R_img_mean", predicted["reliability"][:, 0])
        self._buffer_epoch_metric("R_text_mean", predicted["reliability"][:, 1])
        if self.model.reliability_importance_mode == "relative":
            complete = supervision[:, 0].gt(0.5)
            self._buffer_epoch_metric(
                "relative_supervision_fraction", complete.float().mean()
            )
            if complete.any():
                self._buffer_epoch_metric(
                    "relative_target_std",
                    target[complete, 0].std(unbiased=False),
                )
                self._buffer_epoch_metric(
                    "relative_prediction_std",
                    predicted["reliability"][complete, 0].std(unbiased=False),
                )
        self.log("train/reliability_loss_step", loss.detach(), on_step=True, on_epoch=False)
        return loss

    def _log_adapter_diagnostics(self, diagnostics):
        if not diagnostics:
            return
        for key in (
            "prompt_norm",
            "raw_offset_norm",
            "bounded_offset_norm",
            "raw_offset_prompt_ratio",
            "bounded_offset_prompt_ratio",
            "cap_saturation_ratio",
        ):
            values = torch.cat([item[key].reshape(-1) for item in diagnostics])
            self._buffer_epoch_metric(key, values)

    def _compute_adapter_loss(self, batch):
        image = batch["image"][0]
        output = self.model(
            image,
            batch["text"],
            batch["missing_type"],
            training_phase="adapter",
            return_adapter_diagnostics=True,
        )
        labels = self._labels(batch, image.device)
        loss = F.binary_cross_entropy_with_logits(
            self.mmimdb_classifier(output["cls_feats"]), labels
        )
        self._log_adapter_diagnostics(output["adapter_diagnostics"])
        self._buffer_epoch_metric("task_loss", loss)
        self.log("train/task_loss_step", loss.detach(), on_step=True, on_epoch=False)
        return loss

    def _compute_gate_loss(self, batch):
        if self._phase_for_epoch() != "gate" or not self.training:
            raise RuntimeError("Gate training is allowed only after the phase boundary")
        image = batch["image"][0]
        labels = self._labels(batch, image.device)
        with torch.no_grad():
            state = self.model.prepare_reliability_state(
                image,
                batch["text"],
                batch["missing_type"],
                return_adapter_diagnostics=True,
            )
        self._log_adapter_diagnostics(state["adapter_diagnostics"])
        self._buffer_epoch_metric("R_img_mean", state["reliability"][:, 0])
        self._buffer_epoch_metric("R_text_mean", state["reliability"][:, 1])
        gate_logits = self.model.utility_gate.logits(
            state["reliability"],
            state["availability_mask"],
            state["base_cls_feats"],
        ).reshape(-1)
        g_pred = torch.sigmoid(gate_logits)
        if self.gate_supervision_mode == "direct_task":
            logits = self.mmimdb_classifier(
                self.model.encode_reliability_state(state, g_pred)
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            self._gate_epoch_predictions.append(g_pred.detach().float().cpu())
            self._buffer_epoch_metric("gate_loss", loss)
            self.log(
                "train/gate_loss_step", loss.detach(), on_step=True, on_epoch=False
            )
            return loss

        with torch.no_grad():
            candidate_losses = []
            for value in self.model.gate_candidates:
                gate = torch.full(
                    (image.shape[0], 1), value, device=image.device,
                    dtype=state["reliability"].dtype,
                )
                logits = self.mmimdb_classifier(
                    self.model.encode_reliability_state(state, gate)
                )
                candidate_losses.append(self._sample_loss(logits, labels))
            candidate_losses = torch.stack(candidate_losses, dim=-1)
            best_index = candidate_losses.argmin(dim=-1)
            candidates = torch.as_tensor(
                self.model.gate_candidates,
                device=image.device,
                dtype=state["reliability"].dtype,
            )
            hard_target = candidates.index_select(0, best_index)
            g_target, target_probabilities, candidate_loss_range = soft_gate_target(
                candidate_losses,
                self.model.gate_candidates,
                self.hparams.config.get("gate_target_temperature", 0.025),
            )
            binary_target, endpoint_regret = endpoint_gate_target(
                candidate_losses, self.model.gate_candidates
            )
            target_entropy = -(
                target_probabilities
                * target_probabilities.clamp_min(1e-12).log()
            ).sum(dim=-1) / math.log(len(self.model.gate_candidates))
        self._buffer_epoch_metric("gate_candidate_loss_range", candidate_loss_range)
        self._buffer_epoch_metric("gate_target_entropy", target_entropy)
        self._buffer_epoch_metric("gate_endpoint_regret", endpoint_regret)
        self._buffer_epoch_metric("gate_binary_target_mean", binary_target)
        if self.gate_supervision_mode == "binary_regret":
            g_target = binary_target
            loss = regret_weighted_binary_gate_loss(
                gate_logits, g_target, endpoint_regret
            )
        else:
            loss = F.smooth_l1_loss(g_pred, g_target)
        self._gate_epoch_predictions.append(g_pred.detach().float().cpu())
        self._gate_epoch_targets.append(g_target.detach().float().cpu())
        self._gate_epoch_hard_targets.append(hard_target.detach().float().cpu())
        self._buffer_epoch_metric("gate_loss", loss)
        self.log("train/gate_loss_step", loss.detach(), on_step=True, on_epoch=False)
        return loss

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))
            
        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))
            
        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))              

        return ret

    def _manual_optimizer_step(self, loss, optimizer, batch_idx):
        accumulate = int(self.trainer.accumulate_grad_batches)
        if batch_idx % accumulate == 0:
            optimizer.zero_grad()
        self.manual_backward(loss, optimizer)
        optimizer.step()

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if self.training_mode != "reliability":
            if self.training_mode == "counterfactual":
                target_strength = self.hparams.config.get(
                    "counterfactual_adversary_strength", 1.0
                )
                warmup_steps = self.hparams.config.get(
                    "counterfactual_adversary_warmup_steps", 0
                )
                if warmup_steps > 0:
                    progress = min(
                        1.0, float(self.global_step) / float(warmup_steps)
                    )
                else:
                    progress = 1.0
                current_strength = target_strength * progress
                self.model.counterfactual_proxy.adversary_strength = current_strength
                self.log(
                    "counterfactual/adversary_strength",
                    current_strength,
                    on_step=True,
                    on_epoch=False,
                )
            clip_utils.set_task(self)
            output = self(batch)
            return sum(value for name, value in output.items() if "loss" in name)

        # Lightning 1.1.4 still invokes manual training_step once per configured
        # optimizer.  Execute the complete manual schedule on the first call only.
        if optimizer_idx is None:
            optimizer_idx = 0
        if optimizer_idx != 0:
            return None
        predictor_optimizer, adapter_optimizer, gate_optimizer = self.optimizers()
        phase = self._phase_for_epoch()
        if phase == "adapter":
            reliability_loss = self._compute_reliability_loss(batch)
            self._manual_optimizer_step(
                reliability_loss, predictor_optimizer, batch_idx
            )
            task_loss = self._compute_adapter_loss(batch)
            self._manual_optimizer_step(task_loss, adapter_optimizer, batch_idx)
        else:
            gate_loss = self._compute_gate_loss(batch)
            self._manual_optimizer_step(gate_loss, gate_optimizer, batch_idx)
        return None

    def training_epoch_end(self, outs):
        if self.training_mode != "reliability":
            clip_utils.epoch_wrapup(self, phase="train")
            return
        phase = self._phase_for_epoch()
        for inactive in (
            ("gate_loss",) if phase == "adapter"
            else ("task_loss", "reliability_loss")
        ):
            if inactive not in self._epoch_metric_values:
                self._epoch_metric_values[inactive] = [torch.tensor(0.0)]
        for name, values in self._epoch_metric_values.items():
            self.log("train/{}".format(name), torch.stack(values).mean())
        if self._gate_epoch_predictions:
            predictions = torch.cat(self._gate_epoch_predictions)
            if dist.is_available() and dist.is_initialized():
                gathered_predictions = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_predictions, predictions)
                predictions = torch.cat(gathered_predictions)
            self.log("train/gate_std", predictions.std(unbiased=False))
            self.log("train/gate_mean", predictions.mean())
            quantiles = torch.quantile(
                predictions, torch.tensor([.1, .25, .5, .75, .9])
            )
            for name, value in zip(("p10", "p25", "p50", "p75", "p90"), quantiles):
                self.log("train/gate_{}".format(name), value)
            prediction_std = float(predictions.std(unbiased=False))
            prediction_mean = float(predictions.mean())
            collapsed = prediction_std <= .02
            if self._gate_epoch_targets:
                targets = torch.cat(self._gate_epoch_targets)
                hard_targets = torch.cat(self._gate_epoch_hard_targets)
                if dist.is_available() and dist.is_initialized():
                    gathered_targets = [None for _ in range(dist.get_world_size())]
                    gathered_hard_targets = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(gathered_targets, targets)
                    dist.all_gather_object(gathered_hard_targets, hard_targets)
                    targets = torch.cat(gathered_targets)
                    hard_targets = torch.cat(gathered_hard_targets)
                self.log("train/gate_target_std", targets.std(unbiased=False))
                self.log("train/gate_target_mean", targets.mean())
                self.log("train/gate_mae", (predictions - targets).abs().mean())
                for candidate, label in zip(
                    self.model.gate_candidates, ("0", "025", "05", "075", "1")
                ):
                    self.log(
                        "train/g_star_{}_ratio".format(label),
                        hard_targets.eq(float(candidate)).float().mean(),
                    )
                target_std = float(targets.std(unbiased=False))
                collapsed = target_std >= .05 and prediction_std <= .02
            collapsed = collapsed or (
                (prediction_mean >= .95 or prediction_mean <= .05)
                and prediction_std <= .02
            )
            self.log("train/gate_collapse", float(collapsed))
            if collapsed:
                print("GATE_COLLAPSE")
        self._gate_epoch_predictions = []
        self._gate_epoch_targets = []
        self._gate_epoch_hard_targets = []

    def validation_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        self(batch)

    def validation_epoch_end(self, outs):
        # Lightning 1.1.x may leave the module in training mode for an
        # intra-epoch validation.  Pass the phase explicitly so the
        # ModelCheckpoint monitor always receives val/the_metric.
        clip_utils.epoch_wrapup(self, phase="val")
#         print('missing_img:', self.missing_img_prompt[0,0:3,0:8])
#         print('missing_text:', self.missing_text_prompt[0,0:3,0:8])
#         print('complete:', self.complete_prompt[0,0:3,0:8])

    def test_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        clip_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        if self.training_mode != "reliability":
            return clip_utils.set_schedule(self)
        default_learning_rate = float(self.hparams.config["learning_rate"])
        weight_decay = float(
            self.hparams.config.get(
                "reliability_weight_decay", self.hparams.config["weight_decay"]
            )
        )
        learning_rates = (
            float(
                self.hparams.config.get(
                    "reliability_predictor_lr", default_learning_rate
                )
            ),
            float(
                self.hparams.config.get(
                    "reliability_adapter_lr", default_learning_rate
                )
            ),
            float(
                self.hparams.config.get(
                    "reliability_gate_lr", default_learning_rate
                )
            ),
        )
        optimizers = [
            torch.optim.AdamW(
                self.model.reliability_predictor.parameters(),
                lr=learning_rates[0],
                weight_decay=weight_decay,
            ),
            torch.optim.AdamW(
                self.model.prompt_learner.prompt_adapter.parameters(),
                lr=learning_rates[1],
                weight_decay=weight_decay,
            ),
            torch.optim.AdamW(
                self.model.utility_gate.parameters(),
                lr=learning_rates[2],
                weight_decay=weight_decay,
            ),
        ]
        return optimizers
