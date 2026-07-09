import os
import argparse
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import Unet
from models.C2Net import RNN_Model

from criterions import *
from datasets.dataset_livs import *
from trainer import *


# =========================
# Model Import Utilities
# =========================
def optional_import(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class
    except Exception as exc:
        print(f"[WARN] Could not import {class_name} from {module_name}: {exc}")
        return None


def build_model_registry():
    registry = {
        "Unet": Unet,
        "RNN_Model": RNN_Model,
    }

    optional_models = {
        "PSPNet": ("models.pspnet.pspnet", "PSPNet"),
        "TransUNet": ("models.transunet.transunet", "TransUNet"),
        "SAUNet": ("models.sa_unet.sa_unet", "SAUNet"),
        "R2UNet": ("models.r2_unet.r2_unet", "R2UNet"),
        "RU_Net": ("models.ru_net.ru_net", "RUNet"),
        "LSFPN": ("models.ls_fpn.ls_fpn", "LSFPN"),
        "nnUNet": ("models.nnunet.nnunet", "NNUNet"),
        "UNet3D": ("models.unet3d.unet3d", "UNet3D"),
        "SCUNetPP": ("models.scunet_pp.scunet_pp", "SCUNetPP"),
        "UMamba": ("models.umamba.umamba", "UMamba"),
    }

    for arch_name, (module_name, class_name) in optional_models.items():
        model_class = optional_import(module_name, class_name)
        if model_class is not None:
            registry[arch_name] = model_class

    return registry


MODEL_REGISTRY = build_model_registry()

# =========================
# Argument Parser
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train image segmentation models with unified model interface."
    )

    working_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(working_dir, "results")
    datasets_dir = os.path.join(working_dir, "datasets")

    # Paths
    parser.add_argument("--data-dir", type=str, default=datasets_dir)
    parser.add_argument("--dataset-name", type=str, default="data/LiVS/label")
    parser.add_argument("--logs-dir", type=str, default=os.path.join(results_dir, "logs"))
    parser.add_argument("--weights-dir", type=str, default=os.path.join(results_dir, "weights"))
    parser.add_argument("--predicts-dir", type=str, default=os.path.join(results_dir, "predicts"))
    parser.add_argument("--plots-dir", type=str, default=os.path.join(results_dir, "plots"))

    # Device
    parser.add_argument("--gpu", type=str, default="0")

    # Model
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default="RNN_Model",
        choices=[
            "Unet",
            "RNN_Model",
            "PSPNet",
            "TransUNet",
            "SAUNet",
            "R2UNet",
            "RU_Net",
            "LSFPN",
            "nnUNet",
            "UNet3D",
            "SCUNetPP",
            "UMamba"
        ],
        help="Segmentation model architecture."
    )
    parser.add_argument("--expand_size", type=int, default=1)
    parser.add_argument("--output-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--print-model", action="store_true")

    # Loss
    parser.add_argument(
        "-c",
        "--criterion",
        type=str,
        default="db_criterion",
        choices=["BCEWithLogitsLoss", "db_criterion"]
    )
    parser.add_argument("--pos-weight", type=float, default=None)

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=15)
    parser.add_argument("--accumulate-step", type=int, default=1)

    # Training
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-j", "--workers", type=int, default=8)

    args = parser.parse_args()
    return args


# =========================
# Basic Utilities
# =========================

def create_dirs(args):
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.predicts_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)


def set_seed(seed):
    if seed is None:
        cudnn.benchmark = True
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


def get_device(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA device: {args.gpu}")
    else:
        device = torch.device("cpu")
        print("[INFO] CUDA is not available. Using CPU.")

    return device


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params


def print_model_info(args, model):
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    input_channels = args.expand_size * 2 + 1
    output_channels = args.output_size

    total_params, trainable_params, non_trainable_params = count_parameters(raw_model)

    print("=" * 100)
    print("Model Hyperparameters")
    print("=" * 100)
    print(f"Architecture        : {args.arch}")
    print(f"Input channels      : {input_channels}")
    print(f"Output channels     : {output_channels}")
    print(f"Base channels       : {args.base_channels}")
    print(f"Image size          : {args.img_size}")
    print(f"Expand size         : {args.expand_size}")
    print(f"Input tensor        : [{args.batch_size}, {input_channels}, {args.img_size}, {args.img_size}]")

    if args.arch == "RNN_Model":
        encoder_channels = [
            args.base_channels,
            args.base_channels * 2,
            args.base_channels * 4,
            args.base_channels * 8
        ]
        print(f"Encoder channels    : {encoder_channels}")
        print("Encoder block       : CCRBlock")
        print("Decoder block       : AtrousDecoderBlock")
        print("Decoder kernel      : 3 x 3")
        print("Decoder padding     : 2")
        print("Decoder dilation    : 2")
        print("Normalization       : BatchNorm2d")
        print("Activation          : ReLU")
        print("Upsampling          : Bilinear interpolation")

    print("=" * 100)
    print("Training Hyperparameters")
    print("=" * 100)
    print(f"Epochs              : {args.epochs}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Learning rate       : {args.lr}")
    print(f"Weight decay        : {args.weight_decay}")
    print(f"Optimizer           : AdamW")
    print(f"Scheduler           : ReduceLROnPlateau(mode=max)")
    print(f"LR patience         : {args.lr_patience}")
    print(f"Folds               : {args.folds}")
    print(f"Seed                : {args.seed}")
    print(f"Criterion           : {args.criterion}")
    print(f"Positive weight     : {args.pos_weight}")

    print("=" * 100)
    print("Parameter Statistics")
    print("=" * 100)
    print(f"Total parameters        : {total_params:,}")
    print(f"Trainable parameters    : {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters (M)    : {total_params / 1e6:.4f} M")

    print("=" * 100)
    print("Raw Model Structure")
    print("=" * 100)
    print(raw_model)

    print("=" * 100)
    print("Named Parameters")
    print("=" * 100)
    for name, param in raw_model.named_parameters():
        print(
            f"{name:<70} "
            f"shape={str(list(param.shape)):<25} "
            f"requires_grad={param.requires_grad}"
        )
    print("=" * 100)


# =========================
# Model Construction
# =========================

def instantiate_model(model_class, args):
    """
    Instantiate models with different constructor styles.

    Preferred unified signature:
        Model(in_channels=..., out_channels=..., base_channels=...)

    Compatible signatures:
        Model(input_channels=..., output_channels=..., base_channels=...)
        Model(in_channels, out_channels)
        Model(input_channels, output_channels)
    """
    in_channels = args.expand_size * 2 + 1
    out_channels = args.output_size
    base_channels = args.base_channels

    constructor_trials = [
        dict(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels),
        dict(input_channels=in_channels, output_channels=out_channels, base_channels=base_channels),
        dict(in_channels=in_channels, num_classes=out_channels, base_channels=base_channels),
        dict(input_channels=in_channels, num_classes=out_channels, base_channels=base_channels),
        dict(in_channels=in_channels, out_channels=out_channels),
        dict(input_channels=in_channels, output_channels=out_channels),
        dict(in_channels=in_channels, num_classes=out_channels),
        dict(input_channels=in_channels, num_classes=out_channels),
    ]

    last_error = None

    for kwargs in constructor_trials:
        try:
            return model_class(**kwargs)
        except TypeError as exc:
            last_error = exc

    try:
        return model_class(in_channels, out_channels)
    except TypeError as exc:
        last_error = exc

    raise TypeError(
        f"Failed to instantiate {model_class.__name__}. "
        f"Please wrap the model with a unified constructor. "
        f"Last error: {last_error}"
    )


def get_model(args, device):
    if args.arch not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        raise ImportError(
            f"Model '{args.arch}' is not available. "
            f"Available imported models: {available}. "
            f"Please add its wrapper file under models/ and export the class."
        )

    model_class = MODEL_REGISTRY[args.arch]
    model = instantiate_model(model_class, args)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    input_channels = args.expand_size * 2 + 1
    print(f"[INFO] Model          : {args.arch}")
    print(f"[INFO] Input channels : {input_channels}")
    print(f"[INFO] Output channels: {args.output_size}")
    print(f"[INFO] Base channels  : {args.base_channels}")

    return model


# =========================
# Loss / Optimizer / Scheduler
# =========================

def get_criterion(args, device):
    if args.criterion == "BCEWithLogitsLoss":
        if args.pos_weight is not None:
            pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

    elif args.criterion == "db_criterion":
        try:
            criterion = DB_Criterion(pos_weight=args.pos_weight)
        except TypeError:
            criterion = DB_Criterion()

    else:
        raise ValueError(f"Unsupported criterion: {args.criterion}")

    return criterion


def get_optimizer(args, model):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    return optimizer


def get_scheduler(args, optimizer):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=1e-5
    )
    return scheduler


# =========================
# Data
# =========================

def get_transforms():
    train_transform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
        Normalize([0.], [1.])
    ])

    val_transform = Compose([
        ToTensor(),
        Normalize([0.], [1.])
    ])

    return train_transform, val_transform


def build_dataloaders(args, train_paths, val_paths, train_transform, val_transform):
    trainset = labeledDataset(train_paths, train_transform)
    valset = labeledDataset(val_paths, val_transform)

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    return trainloader, valloader


# =========================
# Forward Sanity Check
# =========================

def run_forward_check(args, model, device):
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.eval()

    input_channels = args.expand_size * 2 + 1

    with torch.no_grad():
        x = torch.randn(
            1,
            input_channels,
            args.img_size,
            args.img_size,
            device=device
        )
        y = raw_model(x)

    if isinstance(y, (tuple, list)):
        y0 = y[0]
    else:
        y0 = y

    print("=" * 100)
    print("Forward Sanity Check")
    print("=" * 100)
    print(f"Input shape : {list(x.shape)}")
    print(f"Output type : {type(y)}")
    print(f"Output shape: {list(y0.shape)}")
    print("=" * 100)


# =========================
# Main
# =========================

def main(args):
    create_dirs(args)
    set_seed(args.seed)

    device = get_device(args)

    print("=" * 100)
    print("Available Imported Models")
    print("=" * 100)
    print(sorted(MODEL_REGISTRY.keys()))
    print("=" * 100)

    data_path = os.path.join(args.data_dir, args.dataset_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    train_transform, val_transform = get_transforms()

    data = split_data(
        data_path,
        nfolds=args.folds,
        expand_size=args.expand_size,
        random_state=args.seed
    )

    for fold in range(args.folds):
        print("=" * 100)
        print(f"Start Fold {fold + 1}/{args.folds}")
        print("=" * 100)

        train_paths, val_paths = data[fold]

        trainloader, valloader = build_dataloaders(
            args=args,
            train_paths=train_paths,
            val_paths=val_paths,
            train_transform=train_transform,
            val_transform=val_transform
        )

        model = get_model(args, device)

        if fold == 0 and args.print_model:
            print_model_info(args, model)
            run_forward_check(args, model, device)

        criterion = get_criterion(args, device)
        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)

        trainer = Trainer(
            args=args,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            scheduler=scheduler,
            fold=fold,
            rnn=None
        )

        trainer.train()

        print("=" * 100)
        print(f"Finished Fold {fold + 1}/{args.folds}")
        print("=" * 100)


if __name__ == "__main__":
    args = parse_args()
    main(args)

