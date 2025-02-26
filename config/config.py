from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    in_image_depth: int
    nb_classes: int
    nb_blocks: int
    block_type: Literal['basic', 'resnet']
    padding: str
    nb_initial_filters: int
    initializer: str
    use_batchnorm: bool
    use_dropout: bool

    def __post_init__(self):
        if self.in_image_depth not in {1, 3}:
            raise ValueError(f"in_image_depth must be either 1 or 3, got {self.in_image_depth}")
        if self.nb_classes <= 2:
            raise ValueError(f"nb_classes must be greater than 2, got {self.nb_classes}")
        if self.nb_blocks <= 2:
            raise ValueError(f"nb_blocks must be greater than 2, got {self.nb_blocks}")
        if not self.block_type.lower() in {'basic', 'resnet'}:
            raise ValueError(f"block_type must be either \'basic\' or \'resnet\', got {self.block_type.lower()}")
        if not self.padding.upper() in {'SAME', 'VALID'}:
            raise ValueError(f"padding must be either \'SAME\' or \'VALID\', got {self.padding.upper()}")
        if self.nb_initial_filters <= 0:
            raise ValueError(f"nb_initial_filters must be greater than 0, got {self.nb_initial_filters}")
        if not self.initializer.lower() in {'he_normal', 'he_uniform'}:
            raise ValueError(f"initializer must be either \'he_normal\' or \'he_uniform\', got {self.initializer.lower()}")


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be greater or equal to 1, got {self.batch_size}")
        if self.learning_rate > 1 or self.learning_rate < 1e-7 :
            raise ValueError(f"Learning rate must be between 0 and 1, got {self.learning_rate}")
        if self.num_epochs < 1:
            raise ValueError(f"Number of epochs must be greater or equal to 1, got {self.num_epochs}")
        
@dataclass
class DatasetConfig:
    train_path: str
    test_path: str


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig

