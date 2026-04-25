import os
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_SIGLIP2_MODEL = os.environ.get(
    "ROBOT_LERF_SIGLIP2_MODEL",
    "google/siglip2-base-patch16-224",
)


class SigLIP2Encoder:
    def __init__(
        self,
        model_name: str = DEFAULT_SIGLIP2_MODEL,
        device: str = "cuda:0",
        prompt_template: str = "This is a photo of {}.",
    ) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "SigLIP2 support requires `transformers` and its tokenizer dependencies. "
                "Install `transformers` and `sentencepiece` in the active environment."
            ) from exc

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.prompt_template = prompt_template
        self.image_batch_size = int(os.environ.get("ROBOT_LERF_SIGLIP2_BATCH_SIZE", "64"))

        model_kwargs = {}
        if self.device.startswith("cuda"):
            model_kwargs["dtype"] = torch.float16
            model_kwargs["attn_implementation"] = "sdpa"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = self._load_model(AutoModel, model_name, model_kwargs).to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    @staticmethod
    def _load_model(auto_model_cls, model_name: str, model_kwargs):
        try:
            return auto_model_cls.from_pretrained(model_name, **model_kwargs)
        except TypeError as exc:
            error_message = str(exc)
            fallback_kwargs = dict(model_kwargs)
            needs_retry = False

            if "unexpected keyword argument 'dtype'" in error_message and "dtype" in fallback_kwargs:
                fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
                needs_retry = True

            if (
                "unexpected keyword argument 'attn_implementation'" in error_message
                and "attn_implementation" in fallback_kwargs
            ):
                fallback_kwargs.pop("attn_implementation")
                needs_retry = True

            if not needs_retry:
                raise

            return auto_model_cls.from_pretrained(model_name, **fallback_kwargs)

    @property
    def model_slug(self) -> str:
        return self.model_name.replace("/", "_").replace("-", "_")

    def encode_image(self, images: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        image_list = self._coerce_image_batch(images)
        features = []
        for start in range(0, len(image_list), self.image_batch_size):
            batch = image_list[start : start + self.image_batch_size]
            processor_inputs = self.processor(images=batch, return_tensors="pt")
            processor_inputs = self._move_inputs(processor_inputs)
            batch_features = self._encode_features("image", processor_inputs)
            features.append(batch_features.detach().cpu())
        return torch.cat(features, dim=0)

    def encode_text(
        self,
        texts: Sequence[str],
        apply_prompt_template: bool = False,
    ) -> torch.Tensor:
        cleaned_texts: List[str] = []
        for text in texts:
            stripped = text.strip()
            cleaned_texts.append(
                self.prompt_template.format(stripped) if apply_prompt_template else stripped
            )
        processor_inputs = self.processor(
            text=cleaned_texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        processor_inputs = self._move_inputs(processor_inputs)
        return self._encode_features("text", processor_inputs)

    def _move_inputs(self, processor_inputs):
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in processor_inputs.items()
        }

    def _encode_features(self, modality: str, processor_inputs) -> torch.Tensor:
        getter = getattr(self.model, f"get_{modality}_features", None)
        with torch.no_grad():
            if getter is not None:
                features = getter(**processor_inputs)
            else:
                outputs = self.model(**processor_inputs)
                features = self._extract_feature_tensor(outputs, modality)
        features = self._extract_feature_tensor(features, modality)
        return F.normalize(features, dim=-1)

    def _extract_feature_tensor(self, outputs, modality: str) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return outputs

        preferred_attrs = [
            f"{modality}_embeds",
            "pooler_output",
            "text_embeds",
            "image_embeds",
            "last_hidden_state",
        ]
        for attr in preferred_attrs:
            value = getattr(outputs, attr, None)
            if torch.is_tensor(value):
                if attr == "last_hidden_state" and value.ndim >= 3:
                    return value.mean(dim=1)
                return value

        if isinstance(outputs, (tuple, list)):
            for value in outputs:
                if torch.is_tensor(value):
                    if value.ndim >= 3:
                        return value.mean(dim=1)
                    return value

        raise TypeError(
            f"Could not extract a feature tensor for modality '{modality}' "
            f"from output type {type(outputs)!r}."
        )

    def _coerce_image_batch(
        self,
        images: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> List[np.ndarray]:
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = [images]
            elif images.ndim == 4:
                images = list(images)
            else:
                raise ValueError(f"Unsupported image tensor shape: {tuple(images.shape)}")

        coerced: List[np.ndarray] = []
        for image in images:
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)
            image = image.detach().cpu()
            if image.ndim != 3:
                raise ValueError(f"Expected 3D image tensor, got shape {tuple(image.shape)}")
            if image.shape[0] in (1, 3):
                image = image.permute(1, 2, 0)
            array = image.numpy().astype(np.float32, copy=True)
            if array.max() > 1.0:
                array = array / 255.0
            coerced.append(np.clip(array, 0.0, 1.0))
        return coerced
