#!/usr/bin/env python3

"""
GitHub-ready training script for the multimodal MRI + report + clinical model.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.utils import resample
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import RobertaModel

from config import TRAINING_EPOCH
from dataset_external import get_train_val_test_Dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal recurrence model.")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--seed_t", type=int, required=True, help="Random seed")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Reserved hidden size flag")
    parser.add_argument("--dropout", type=float, required=True, help="Transformer fusion dropout")
    parser.add_argument("--lr_image", type=float, required=True, help="Learning rate for image branch")
    parser.add_argument("--lr_report", type=float, required=True, help="Learning rate for report branch")
    parser.add_argument("--lr_total", type=float, required=True, help="Learning rate for fusion and head")
    parser.add_argument("--clin_hidden_dim", type=int, default=256, help="Reserved clinical hidden size flag")
    parser.add_argument("--clin_layers", type=int, default=2, help="Reserved clinical layer count flag")
    parser.add_argument("--lr_clin", type=float, default=None, help="Learning rate for clinical branch")
    parser.add_argument("--radiobert_path", type=str, required=True, help="Path to pretrained RadioBERT directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCH, help="Training epochs")
    parser.add_argument("--step_size", type=int, default=5, help="StepLR step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="StepLR gamma")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Optimizer weight decay")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda")
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out, inplace=True)


class ResNet3D(nn.Module):
    def __init__(self, block: nn.Module, layers: List[int]):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, block: nn.Module, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def resnet18_3d() -> ResNet3D:
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2])


class RadioLOGIC(nn.Module):
    def __init__(self, radiobert_path: str):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(radiobert_path, add_pooling_layer=False)

    def forward(self, input_id: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_id, attention_mask=attention_mask, return_dict=False)
        return outputs[0][:, 0]


class ImageNetBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18_3d()
        self.projection = nn.Linear(512, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.resnet(x))


class ReportNet(nn.Module):
    def __init__(self, radiobert_path: str):
        super().__init__()
        self.encoder = RadioLOGIC(radiobert_path)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.projection = nn.Sequential(nn.Linear(768, 256), nn.ReLU(inplace=True))

    def forward(self, input_id: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(input_id, attention_mask))


class ClinicalNet(nn.Module):
    def __init__(self, dropout_prob: float = 0.3):
        super().__init__()
        input_dim = 25 * 12 + 10
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        return self.fc(x.view(x.size(0), -1))


class TransformerFusion(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


class MultiModalModel(nn.Module):
    def __init__(self, radiobert_path: str, dropout_prob: float = 0.4):
        super().__init__()
        self.image_branch = ImageNetBranch()
        self.report_branch = ReportNet(radiobert_path)
        self.clinical_branch = ClinicalNet(dropout_prob=0.2)
        self.fusion_stage1 = TransformerFusion(embed_dim=256, dropout=dropout_prob)
        self.fusion_stage2 = TransformerFusion(embed_dim=256, dropout=dropout_prob)
        self.classifier = nn.Linear(256, 1)

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        input_id: torch.Tensor,
        attention_mask: torch.Tensor,
        clinical_feature: torch.Tensor,
    ) -> torch.Tensor:
        image_input = torch.cat((image1, image2), dim=1)
        image_features = self.image_branch(image_input)
        report_features = self.report_branch(input_id, attention_mask)
        clinical_features = self.clinical_branch(clinical_feature)
        tokens_stage1 = torch.stack([image_features, report_features], dim=1)
        fused_ir = self.fusion_stage1(tokens_stage1).mean(dim=1)
        tokens_stage2 = torch.stack([fused_ir, clinical_features], dim=1)
        fused_final = self.fusion_stage2(tokens_stage2).mean(dim=1)
        return self.classifier(fused_final)


def cox_loss(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    score = y_pred.view(-1)
    time = y_true[:, 0]
    event = y_true[:, 1].bool()

    order = torch.argsort(time, descending=True)
    time = time[order]
    event = event[order]
    score = score[order]

    exp_score = torch.exp(score)
    cumsum_exp = torch.cumsum(exp_score, dim=0)

    if event.sum() == 0:
        return score.sum() * 0.0

    unique_times, inv_idx = torch.unique_consecutive(time[event], return_inverse=True)
    is_last_in_block = torch.ones_like(time, dtype=torch.bool)
    is_last_in_block[:-1] = time[:-1] != time[1:]
    block_last_indices = torch.nonzero(is_last_in_block, as_tuple=False).view(-1)
    block_times = time[block_last_indices]

    idx_map = {}
    j = 0
    for bi, t_val in enumerate(block_times):
        while j < len(unique_times) and unique_times[j] == t_val:
            idx_map[int(j)] = block_last_indices[bi].item()
            j += 1
        if j >= len(unique_times):
            break

    total_loglik = torch.zeros((), dtype=score.dtype, device=score.device)
    num_events = 0
    event_indices = torch.nonzero(event, as_tuple=False).view(-1)

    for g in range(len(unique_times)):
        g_mask = inv_idx == g
        if not torch.any(g_mask):
            continue
        d_t = event_indices[g_mask]
        m = d_t.numel()
        num_events += m
        risk_sum = cumsum_exp[idx_map[int(g)]]
        tied_score_sum = torch.sum(score[d_t])
        tied_exp_sum = torch.sum(exp_score[d_t])
        frac = torch.arange(m, dtype=score.dtype, device=score.device) / max(m, 1)
        denom_terms = torch.clamp(risk_sum - frac * tied_exp_sum, min=eps)
        total_loglik = total_loglik + tied_score_sum - torch.sum(torch.log(denom_terms))

    if num_events == 0:
        return score.sum() * 0.0

    return (-total_loglik / float(num_events)) + 0.0 * score.sum()


def concordance_index_train(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    time = y_true[:, 0]
    event = y_true[:, 1].bool()
    pred = y_pred.view(-1)
    t_i = time[:, None]
    t_j = time[None, :]
    e_i = event[:, None]
    p_i = pred[:, None]
    p_j = pred[None, :]
    comparable = (t_i < t_j) & e_i
    denom = comparable.float().sum()
    if denom == 0:
        return pred.sum() * 0.0
    score = (p_i > p_j).float() + 0.5 * (p_i == p_j).float()
    return (score * comparable.float()).sum() / denom


def concordance_index_eval(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    actual_durations = y_true[:, 0].detach().cpu().numpy()
    predicted_scores = y_pred.detach().cpu().numpy()
    event_observed = y_true[:, 1].detach().cpu().numpy()
    if np.isnan(predicted_scores).any():
        return 0.5
    return concordance_index(
        event_times=actual_durations,
        predicted_scores=predicted_scores,
        event_observed=event_observed,
    )


def cal_loss(y: torch.Tensor, pred: torch.Tensor) -> Dict[int, torch.Tensor]:
    loss_dict = {}
    for target_class in range(int(y.shape[1] / 2)):
        loss_dict[target_class] = cox_loss(y[:, target_class * 2:(target_class + 1) * 2], pred)
    return loss_dict


def cal_ci_test(y: torch.Tensor, pred: torch.Tensor) -> Dict[int, float]:
    ci_dict = {}
    for target_class in range(int(y.shape[1] / 2)):
        ci_dict[target_class] = concordance_index_eval(y[:, target_class * 2:(target_class + 1) * 2], -pred)
    return ci_dict


def unpack_batch(data: dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image1 = data["image1"].to(device)
    image2 = data["image2"].to(device)
    clin_features = data["clinical_features"].to(device)
    label_cls = data["label_cls"].to(device)
    mask = data["report_code"]["attention_mask"][:, 0, :].long().to(device)
    input_id = data["report_code"]["input_ids"][:, 0, :].long().to(device)
    return image1, image2, input_id, mask, clin_features, label_cls


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0
    for data in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", position=0, leave=True):
        if data is None:
            continue
        image1, image2, input_id, mask, clin_features, label_cls = unpack_batch(data, device)
        optimizer.zero_grad()
        outputs = model(image1, image2, input_id, mask, clin_features)
        loss = sum(cal_loss(label_cls, outputs).values())
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()
        steps += 1
    return running_loss / max(steps, 1)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader, device: torch.device) -> Tuple[Dict[int, torch.Tensor], Dict[int, float], torch.Tensor]:
    pred_all = None
    y_all = None
    model.eval()
    for data in dataloader:
        if data is None:
            continue
        image1, image2, input_id, mask, clin_features, label_cls = unpack_batch(data, device)
        pred = model(image1, image2, input_id, mask, clin_features)
        if pred_all is None:
            pred_all = pred
            y_all = label_cls
        else:
            pred_all = torch.cat([pred_all, pred], dim=0)
            y_all = torch.cat([y_all, label_cls], dim=0)

    if pred_all is None or y_all is None:
        raise RuntimeError("No valid samples were loaded. Check dataset paths and JSON metadata.")

    return cal_loss(y_all, pred_all), cal_ci_test(y_all, pred_all), pred_all


def main() -> None:
    args = parse_args()
    setup_seed(args.seed_t)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / f"{args.name}.pth"
    run_config_path = output_dir / f"{args.name}_config.json"

    train_dataloader, val_dataloader, test_dataloader, _ = get_train_val_test_Dataloaders()

    model = MultiModalModel(
        radiobert_path=args.radiobert_path,
        dropout_prob=args.dropout,
    ).to(device)

    lr_clin = args.lr_clin if args.lr_clin is not None else args.lr_total
    param_groups = [
        {"params": model.image_branch.parameters(), "lr": args.lr_image},
        {"params": model.clinical_branch.parameters(), "lr": lr_clin},
        {"params": model.report_branch.projection.parameters(), "lr": args.lr_report},
        {"params": model.fusion_stage1.parameters(), "lr": args.lr_total},
        {"params": model.fusion_stage2.parameters(), "lr": args.lr_total},
        {"params": model.classifier.parameters(), "lr": args.lr_total},
    ]

    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    with run_config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    best_val_ci = float("-inf")
    print(f"seed: {args.seed_t}")
    print(f"device: {device}")
    print(f"checkpoint: {best_model_path}")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device, epoch, args.epochs)
        scheduler.step()

        train_loss_dict, train_ci_dict, _ = evaluate(model, train_dataloader, device)
        val_loss_dict, val_ci_dict, _ = evaluate(model, val_dataloader, device)
        test_loss_dict, test_ci_dict, _ = evaluate(model, test_dataloader, device)

        train_ci = sum(train_ci_dict.values())
        val_ci = sum(val_ci_dict.values())
        test_ci = sum(test_ci_dict.values())

        print(f"epoch={epoch + 1} train_loss={train_loss:.6f}")
        print(f"train_loss_dict={train_loss_dict} val_loss_dict={val_loss_dict} test_loss_dict={test_loss_dict}")
        print(f"train_c_index={train_ci} val_c_index={val_ci} test_c_index={test_ci}")

        if val_ci > best_val_ci:
            best_val_ci = val_ci
            torch.save(model.state_dict(), best_model_path)
            print(f"saved best model with val_c_index={best_val_ci:.4f}")

    print(f"best validation c-index: {best_val_ci:.4f}")


if __name__ == "__main__":
    main()
