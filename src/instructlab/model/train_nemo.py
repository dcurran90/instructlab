# SPDX-License-Identifier: Apache-2.0

# Standard
from glob import glob
from pathlib import Path
import os
import shutil

# Third Party
import click
import torch

# First Party
from instructlab import utils
from nemo.core.config import hydra_runner

from nemo.utils.nemo_logging import Logger as _Logger
from nemo.utils.nemo_logging import LogMode as logging_mode
from omegaconf.omegaconf import OmegaConf

logging = _Logger()

class TorchDeviceParam(click.ParamType):
    """Parse and convert device string

    Returns a torch.device object:
    - type is one of 'cpu', 'cuda', 'hpu'
    - index is None or device index (e.g. 0 for first GPU)
    """

    name = "deviceinfo"
    supported_devices = {"cuda", "cpu", "hpu"}

    def convert(self, value, param, ctx) -> "torch.device":
        # pylint: disable=C0415
        # Function local import, import torch can take more than a second
        # Third Party
        import torch

        if not isinstance(value, torch.device):
            try:
                device = torch.device(value)
            except RuntimeError as e:
                self.fail(str(e), param, ctx)

        if device.type not in self.supported_devices:
            supported = ", ".join(repr(s) for s in sorted(self.supported_devices))
            self.fail(
                f"Unsupported device type '{device.type}'. Only devices "
                f"types {supported}, and indexed device strings like 'cuda:0' "
                "are supported for now.",
                param,
                ctx,
            )

        # Detect CUDA/ROCm device
        if device.type == "cuda":
            if not torch.cuda.is_available():
                self.fail(
                    f"{value}: Torch has no CUDA/ROCm support or could not detect "
                    "a compatible device.",
                    param,
                    ctx,
                )
            # map unqualified 'cuda' to current device
            if device.index is None:
                device = torch.device(device.type, torch.cuda.current_device())

        if device.type == "hpu":
            click.secho(
                "WARNING: HPU support is experimental, unstable, and not "
                "optimized, yet.",
                fg="red",
                bold=True,
            )

        return device


TORCH_DEVICE = TorchDeviceParam()


@click.command()
@click.option("--data-dir", help="Base directory where data is stored.", default=None)
@click.option(
    "--input-dir",
    type=click.Path(),
    show_default=True,  # TODO: set to None and change help message
    help="Path to generated files to use as input.",
)
@click.option(
    "--skip-preprocessing",
    is_flag=True,
)
@click.option(
    "--tokenizer-dir",
    help="Base directory where tokenizer is stored.",
    default=None,
    show_default=True,
)
@click.option(
    "--gguf-model-path",
    help="Local directory where gguf model is stored.",
    default=None,
    show_default=True,
)
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    default="instructlab/merlinite-7b-lab",
    show_default=True,
)
@click.option("--iters", help="Number of iterations to train LoRA.", default=100)
@click.option(
    "--local",
    is_flag=True,
    help="Whether or not `model_dir` is remote from HuggingFace.",
)
@click.option(
    "-sq",
    "--skip-quantize",
    is_flag=True,
    help="Whether to skip quantization while converting to MLX. This parameter will be ignored if --gguf-model-path and --tokenizer-dir are specified.",
)
@click.option(
    "--num-epochs",
    type=click.INT,
    default=1,  # TODO: change this to a more reasonable default
    show_default=True,
    help="The number of times the training data is passed through the training algorithm. Please note that this value is used on Linux platforms only.",
)
@click.option(
    "--device",
    type=TORCH_DEVICE,
    show_default=True,
    default="cpu",
    help=(
        "PyTorch device for Linux training (default: 'cpu'). Use 'cuda' "
        "for NVidia CUDA / AMD ROCm GPU, 'cuda:0' for first GPU."
    ),
)
@click.option(
    "--4-bit-quant",
    "four_bit_quant",
    is_flag=True,
    show_default=True,
    default=False,
    # TODO: hidden option until llamacpp_convert_to_gguf.py supports
    # quantized models, https://github.com/instructlab/instructlab/issues/579
    hidden=True,
    help=(
        "Use BitsAndBytes for 4-bit quantization "
        "(reduces GPU VRAM usage and may slow down training)"
    ),
)
@click.option(
    "--model-name",
    default="instructlab/merlinite-7b-lab",
    show_default=True,
    help="model name to use in training",
)
@click.pass_context
#@hydra_runner(config_path="../config", config_name="megatron_gpt_finetuning_config")
def train_nemo(
    ctx,
    data_dir,
    input_dir,
    skip_preprocessing,
    tokenizer_dir,
    gguf_model_path,
    model_dir,
    iters,
    local,
    skip_quantize,
    num_epochs,
    device: "torch.device",
    four_bit_quant: bool,
    model_name: str,
):
    """
    Takes synthetic data generated locally with `ilab generate` and the previous model and learns a new model using the MLX API.
    On success, writes newly learned model to {model_dir}/mlx_model, which is where `chatmlx` will look for a model.
    """
    #from hydra import compose, initialize
    import hydra
    from omegaconf import OmegaConf

    hydra.initialize(config_path="../config", job_name="megatron_gpt_finetuning")
    
    # Load the configuration
    cfg = hydra.compose(config_name="megatron_gpt_finetuning")

    # pylint: disable=import-outside-toplevel
    if not utils.is_macos_with_m_chip():
        # Local
        # from ..llamacpp.llamacpp_convert_to_gguf import convert_llama_to_gguf
        # from ..train.linux_train import linux_train
        from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
        from nemo.utils.exp_manager import exp_manager
        from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP



        logging.info("\n\n************** Experiment configuration ***********")
        logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

        trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()
        exp_manager(trainer, cfg.exp_manager)

        model_cfg = MegatronGPTSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
        model = MegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
        peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]

        if cfg.model.peft.restore_from_path is not None:
            # initialize peft weights from a checkpoint instead of randomly
            # This is not the same as resume training because optimizer states are not restored.
            logging.info("PEFT Weights will be loaded from", cfg.model.peft.restore_from_path)
            model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls(model_cfg))
        elif peft_cfg_cls is not None:
            logging.info("Adding adapter weights to the model for PEFT")
            model.add_adapter(peft_cfg_cls(model_cfg))
        else:
           logging.info(f"Running full finetuning since no peft scheme is given.\n{model.summarize()}")

        trainer.fit(model)

