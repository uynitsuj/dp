import dataclasses
import pathlib
import abc
import difflib
from typing import Literal
from typing_extensions import override

import tyro

from dp.util.args import SharedConfig, DatasetConfig, TrainerConfig, LoggingConfig, PolicyConfig, VisionEncoderConfig, PreferenceLearningConfig, ModelConfig, OptimizerConfig
from dp.util import transforms as _transforms
from dp.util.lerobot_conv_utils import convert_dataset_parallel

@dataclasses.dataclass(frozen=True)
class LeRobotDatasetConfigFactory(DatasetConfig):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DatasetConfig | None] = None

    @abc.abstractmethod
    def create(self) -> DatasetConfig:
        """Create a data config."""

    def create_base_config(self) -> DatasetConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None

        dataset_root = convert_dataset_parallel(repo_id=repo_id)

        return dataclasses.replace(
            self.base_config or LeRobotDatasetConfigFactory(),
            repo_id=repo_id,
            dataset_root=str(dataset_root),
            # norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotXmiRbyDatasetConfig(LeRobotDatasetConfigFactory):
    """
    This config is used to configure transforms for the XMI RBY bimanual robot dataset.
    
    The XMI data uses end-effector poses with 6D rotation representation:
    - State format: [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper] = 20D
    - Three camera views: left exterior, right exterior, and top
    - Actions are delta end-effector poses with absolute gripper positions
    """
    
    # If provided, will be injected into the input data if the "prompt" key is not present.
    # default_prompt: str | None = None
    retarget_mode: Literal["20D-relative", "20D-intergripper-relative", "29D-relative", "29D-intergripper-relative"] = "29D-relative"
    use_top_camera: bool = True
    
    @override
    def create(self) -> DatasetConfig:

        # Data transforms using XMI RBY policy transforms
        data_transforms = _transforms.Group()

        # XMI data uses delta actions for rotations/positions, but absolute gripper positions
        # The conversion script already produces the correct format, but we may need delta conversion
        # for the rotations and positions (indices 0:6, 6:9, 10:16, 16:19) while keeping
        # grippers absolute (indices 9, 19)
        if "20D" in self.retarget_mode:
            delta_action_mask = _transforms.make_bool_mask(
                9, -1,  # left: 6d_rot (delta), 3d_pos (delta), gripper (absolute)
                9, -1   # right: 6d_rot (delta), 3d_pos (delta), gripper (absolute) 
            )
        elif "29D" in self.retarget_mode:
            delta_action_mask = _transforms.make_bool_mask(
                9, -1,  # left: 6d_rot (delta), 3d_pos (delta), gripper (absolute)
                9, -1,   # right: 6d_rot (delta), 3d_pos (delta), gripper (absolute) 
                9,   # head: 6d_rot (delta), 3d_pos (delta)
            )

        if self.retarget_mode == "20D-relative" or self.retarget_mode == "29D-relative":
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        elif self.retarget_mode == "20D-intergripper-relative" or self.retarget_mode == "29D-intergripper-relative":
            data_transforms = data_transforms.push(
                inputs=[_transforms.Bimanual_InterGripperProprio_DeltaActions(delta_action_mask, action_dim=20 if "20D" in self.retarget_mode else 29)],
                outputs=[_transforms.Bimanual_InterGripperProprio_AbsoluteActions(delta_action_mask, action_dim=20 if "20D" in self.retarget_mode else 29)],
            )

        return dataclasses.replace(
            self.create_base_config(),
            data_transforms=data_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotYamDatasetConfig(LeRobotDatasetConfigFactory):
    """
    This config is used to configure transforms for the YAM bimanual robot dataset.
    The YAM data can use absolute joint positions or absolute cartesian positions:
    - State format: [left_6_joints, left_1_gripper, right_6_joints, right_1_gripper] = 14D
    or
    - State format: [left_6d_rot, left_3d_pos, left_1d_gripper, right_6d_rot, right_3d_pos, right_1d_gripper] = 20D
    - Three camera views: left exterior, right exterior, and top
    - Actions are absolute joint positions or absolute cartesian positions
    """
    
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    action_space: Literal["joint", "cartesian"] = "joint"

    @override
    def create(self) -> DatasetConfig:

        # model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        if self.action_space == "joint":
            robot_action_dim = 14
        elif self.action_space == "cartesian":
            robot_action_dim = 20
        else:
            raise ValueError(f"Invalid action space: {self.action_space}")

        # Data transforms using YAM policy transforms
        data_transforms = _transforms.Group(
            inputs=[yam_policy.YamInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[yam_policy.YamOutputs(robot_action_dim=robot_action_dim)],
        )

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(),
            data_transforms=data_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "dp"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    # model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    # weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    # optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    # ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    # freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    dataset_cfg: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)

    # Model configuration
    model_cfg: ModelConfig = dataclasses.field(default_factory=ModelConfig)

    # Optimizer configuration
    optimizer_cfg: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)

    # Shared configuration
    shared_cfg: SharedConfig = dataclasses.field(default_factory=SharedConfig)

    # Logging configuration 
    logging_cfg: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)

    # trainer configuration
    trainer_cfg: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)

    # train or eval 
    train : bool = True

    # distributed training on multiple GPUs # TODO: verify functionality
    distributed: bool = False

    # number of distributed processes (required by torch distributed)
    world_size: int = 1

    # local rank of the process (required by torch distributed)
    local_rank: int = -1

    # distributed training on the optimizer (required by torch distributed)
    dist_on_itp: bool = False

    # distributed training url (required by torch distributed)
    dist_url: str = 'env://'

    # device to use for training / testing (required by torch distributed)
    device : str = "cuda"

    # load config. instead of using command line arguments, load from a config file
    # load_config: Optional[str] = None

    # Base directory for config assets (e.g., norm stats).
    # assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    # checkpoint_base_dir: str = "./checkpoints"
    # checkpoint_base_dir: str = "/home/justinyu/checkpoints" # TODO: implement functionality

    # Random seed that will be used by random generators during training.
    # seed: int = 42
    # Global batch size.
    # batch_size: int = 128
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    # num_workers: int = 20
    # Number of train steps (batches) to run.
    # num_train_epochs: int = 300

    # How often (in steps) to log training metrics.
    # log_interval: int = 100
    # How often (in steps) to save checkpoints.
    # save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    # keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists. # TODO: implement functionality
    overwrite: bool = False
    # If true, will resume training from the last checkpoint. # TODO: implement functionality
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True # TODO: implement functionality

    # Used to pass metadata to the policy server.
    # policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    # fsdp_devices: int = 1

    # S3 path for saving checkpoints (optional). If set, checkpoints will be saved to S3. # TODO: implement functionality
    s3_checkpoint_path: str | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


_CONFIGS = [
    TrainConfig(
        name="dp_xmi_rby",

        dataset_cfg=LeRobotXmiRbyDatasetConfig(
            repo_id="uynitsuj/soup_can_in_domain_xmi_data_center_cropped_20250818",
            # default_prompt="pick up the soup can and place it in the bin",

            retarget_mode="29D-relative",
            use_top_camera=True,
        ),

        shared_cfg=SharedConfig(
            batch_size=128,
            num_pred_steps=40,
            num_cameras=3,
            camera_keys = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]
        ),
        trainer_cfg=TrainerConfig(
            epochs=300,
        ),
        logging_cfg=LoggingConfig(
            log_name="250819_1340", # TODO: unify with exp_name
            output_dir="/nfs_us/justinyu/dp",
        ),
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

if __name__ == "__main__": 
    args = tyro.cli(TrainConfig)
    dict_args = dataclasses.asdict(args)

