"""Explicit model and stage settings stored alongside each checkpoint."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import yaml


@dataclass
class ModelConfig:
    base_model: str = "/path/to/model"
    local_files_only: bool = True
    rank: int = 8
    dropout: float = 0.05
    beta_max: float = 8.0
    context_mix: float = 0.5
    centered_residual: bool = True
    protected: bool = True
    interaction: bool = True
    max_tokens: int = 1536
    control_tokens: list[str] = field(
        default_factory=lambda: ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
    )
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    def validate(self):
        if (
            not isinstance(self.control_tokens, list)
            or len(self.control_tokens) != 3
            or not all(isinstance(token, str) and token for token in self.control_tokens)
            or len(set(self.control_tokens)) != 3
        ):
            raise ValueError("Provide three distinct nonempty control token spellings")
        if (
            not isinstance(self.target_modules, list)
            or not self.target_modules
            or not all(isinstance(name, str) and name for name in self.target_modules)
        ):
            raise ValueError("Provide the model's LoRA target module names")
        if not self.base_model or self.rank < 1 or self.max_tokens < 1:
            raise ValueError("Invalid model path, LoRA rank or sequence limit")
        if not 0 <= self.dropout < 1 or not 0 <= self.context_mix <= 1:
            raise ValueError("Invalid dropout or evidence mixture")
        if not math.isfinite(self.beta_max) or self.beta_max <= 0:
            raise ValueError("The evidence budget ceiling needs to be positive")


@dataclass
class TrainingConfig:
    stage: str
    model: ModelConfig = field(default_factory=ModelConfig)
    epochs: int = 2
    lr: float = 5e-5
    group_batch: int = 2
    weight_decay: float = 0.01
    invariance_weight: float = 0.0075
    counterfactual_weight: float = 0.0075
    authority_weight: float = 0.025
    margin_weight: float = 0.025
    margin_slack: float = 1.0
    adversarial_weight: float = 0.25
    adversarial_invariance_weight: float = 0.0075
    gradient_checkpointing: bool = True

    def validate(self):
        self.model.validate()
        if self.stage not in {"stage1", "stage2", "authentication"}:
            raise ValueError("Unknown training stage")
        if self.epochs < 1 or self.group_batch < 1 or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Epochs, group batch and learning rate need to be positive")
        for name in (
            "weight_decay",
            "invariance_weight",
            "counterfactual_weight",
            "authority_weight",
            "margin_weight",
            "adversarial_weight",
            "adversarial_invariance_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if not math.isfinite(self.margin_slack) or self.margin_slack <= 0:
            raise ValueError("The policy margin slack needs to be positive")
        expected = (1.0, False, False) if self.stage == "stage1" else (0.5, True, True)
        actual = (self.model.context_mix, self.model.protected, self.model.interaction)
        if actual != expected or not self.model.centered_residual:
            raise ValueError("Model settings do not match the stage architecture")

    def to_dict(self):
        return asdict(self)


def load_config(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("The configuration needs to be a mapping")
    config = TrainingConfig(**{**data, "model": ModelConfig(**data.get("model", {}))})
    config.validate()
    return config
