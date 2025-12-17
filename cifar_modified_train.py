from __future__ import annotations

# cifar_modified_train.py

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from cifar_loader import get_cifar10_loaders, pick_device
from cifar_modified_nnet import CifarModifiedNet
from cifar_video_emitter import CifarVideoEmitter  # the bridge that calls render_training_frame


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "auto"

    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 2

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-5

    conv1_out: int = 24
    conv2_out: int = 48
    kernel_size: int = 3
    dropout: float = 0.1

    log_every: int = 100
    save_path: str = "./checkpoints/cifar_modified.pt"

    # video knobs
    reference_image_path: str = "reference_cat.png"
    frames_out_dir: str = "training_video_frames"
    emit_every_steps: int = 1         # every training step (set to 5/10 if too many frames)

    # "snapshot" (one frame per tick) or "nudge" (many frames per tick)
    emit_mode: str = "snapshot"

    # Only used if emit_mode == "nudge"
    nudge_mode: str = "out_in"        # "out" | "out_in" | "tap"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_reference_tensor(path: str, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (32, 32):
        raise ValueError(f"Reference must be 32x32, got {img.size}")
    x = T.ToTensor()(img).unsqueeze(0)  # (1,3,32,32)
    return x.to(device)


@torch.no_grad()
def evaluate(model, dl, device: str):
    model.set_video(False, emit=None)   # IMPORTANT: no emitting during eval
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss_sum += float(loss) * xb.size(0)
        correct += int((logits.argmax(1) == yb).sum())
        total += xb.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = pick_device(cfg.device)
    print("Device:", device)

    loaders = get_cifar10_loaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
    )

    model = CifarModifiedNet(
        conv1_out=cfg.conv1_out,
        conv2_out=cfg.conv2_out,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Video emitter + reference tensor
    emitter = CifarVideoEmitter(
        reference_image_path=cfg.reference_image_path,
        output_directory=cfg.frames_out_dir,
    )
    x_ref = load_reference_tensor(cfg.reference_image_path, device=device)

    steps_per_epoch = len(loaders.train)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        correct_run = 0
        total_run = 0

        for step, (xb, yb) in enumerate(loaders.train, start=1):
            xb, yb = xb.to(device), yb.to(device)

            # Train step (video OFF)
            model.set_video(False, emit=None)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)

            pred = logits.argmax(1)
            correct_run += int((pred == yb).sum())
            total_run += int(yb.numel())
            train_acc = correct_run / max(total_run, 1)

            loss = F.cross_entropy(logits, yb)

            

            loss.backward()
            opt.step()

            running_loss += float(loss) * xb.size(0)
            seen += xb.size(0)

            # Optional logging
            if step % cfg.log_every == 0:
                train_loss = running_loss / max(seen, 1)
                test_loss, test_acc = evaluate(model, loaders.test, device)
                print(
                    f"epoch {epoch:02d} step {step:04d} | "
                    f"train_loss {train_loss:.4f} | "
                    f"test_loss {test_loss:.4f} | test_acc {test_acc*100:.2f}%"
                )

            # Video emit step (reference only)
            if cfg.emit_every_steps > 0 and (step % cfg.emit_every_steps == 0):
                emitter.set_header(
                    accuracy=train_acc,
                    epoch=epoch,
                    epochs=cfg.epochs,
                    progress=step / steps_per_epoch,
                )

                model.eval()
                model.set_video(
                    True,
                    emit=emitter.emit,
                    ref_index=0,
                    emit_mode=cfg.emit_mode,
                    nudge_mode=cfg.nudge_mode,
                )
                with torch.no_grad():
                    _ = model(x_ref)

        test_loss, test_acc = evaluate(model, loaders.test, device)
        print(f"[EPOCH END] epoch {epoch:02d} | test_loss {test_loss:.4f} | test_acc {test_acc*100:.2f}%")

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "train_config": cfg.__dict__}, cfg.save_path)
    print("Saved checkpoint:", cfg.save_path)


if __name__ == "__main__":
    main()
