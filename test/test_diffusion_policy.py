# CUDA_VISIBLE_DEVICES=4 python test/test_diffusion_policy.py --dataset-cfg.dataset-root /home/yujustin/dataset/dp_gs/sim_coffee_maker/successes_033125_1619/ --logging-cfg.log-name 250331_2233 --logging-cfg.output-dir /home/yujustin/dp_gs/ --shared-cfg.no-use-delta-action --shared-cfg.seq-length 4 --shared-cfg.num-pred-steps 16 --dataset-cfg.subsample-steps 2 --trainer-cfg.epochs 300 --dataset-cfg.is-sim-data --shared-cfg.image_size 448 --optimizer-cfg.lr 6e-4 --model-cfg.policy-cfg.down-dims 256 256 512 1024 --shared-cfg.batch-size 64 --model-cfg.policy-cfg.pred_right_only --model-cfg.policy-cfg.diffusion-model-type transformer
import torch
from dp_gs.policy.model import DiffusionPolicy, SimplePolicy, Dinov2DiscretePolicy
from dp_gs.util.args import ExperimentConfig
import tyro


def test_diffusion_policy(args: ExperimentConfig):
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the policy model directly
    if args.model_cfg.policy_type == "diffusion":
        policy = DiffusionPolicy
    elif args.model_cfg.policy_type == "discrete":
        policy = Dinov2DiscretePolicy
    else:
        policy = SimplePolicy

    model = policy(
        shared_cfg=args.shared_cfg,
        model_cfg=args.model_cfg
    ).to(device)
    model.train()

    # Create a dummy batch for testing
    if args.model_cfg.policy_cfg.pred_action_only or args.model_cfg.policy_cfg.pred_right_only:
        ratio = 2 
    else:
        ratio = 1
    batch = {
        "observation": torch.randn(1, model.obs_horizon, 
                                    model.num_cameras, 3, 
                                    args.shared_cfg.image_size, args.shared_cfg.image_size).to(device),
        "proprio": torch.randn(1, model.obs_horizon, 20).to(device),
        "action": torch.randn(1, model.obs_horizon, model.action_horizon, model.action_dim * ratio).to(device)  # Add dummy actions
    }

    # Check if parameters are frozen
    frozen_params = [name for name, param in model.named_parameters() if not param.requires_grad]
    if frozen_params:
        print("Frozen Parameters:", frozen_params)
    else:
        print("No frozen parameters.")

    # Perform forward pass
    forward_loss = model.forward(batch)
    forward_loss.backward()
    print("Forward Loss:", forward_loss)

    # Check for parameters with gradients
    params_with_grad = [name for name, param in model.named_parameters() if param.grad is not None]
    params_without_grad = [name for name, param in model.named_parameters() if param.grad is None]
    if params_without_grad:
        print("Parameters with gradients:", params_with_grad)
        print("Parameters without gradients:", params_without_grad)
    else:
        print("All parameters have gradients.")

    # Perform inference
    pred_action = model.forward_inference(batch)
    print("Predicted Action:", pred_action)


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    test_diffusion_policy(args) 