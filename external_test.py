#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GitHub-ready survival evaluation script for multimodal MRI + report + clinical model.

Main improvements over the original script:
1. Removed duplicated imports and repeated evaluation blocks.
2. Replaced hard-coded paths with CLI arguments.
3. Encapsulated logic into reusable functions.
4. Unified internal / external evaluation workflow.
5. Improved readability and maintainability for GitHub release.

Required project files:
- config.py
- dataset_external.py
- transforms.py

Example:
python github_ready_survival_eval.py \
    --name exp001 \
    --seed_t 42 \
    --hidden_dim 256 \
    --dropout 0.4 \
    --lr_image 1e-5 \
    --lr_report 1e-5 \
    --lr_total 1e-5 \
    --model_paths /path/to/model.pth \
    --radiobert_path /path/to/radiobert \
    --internal_json /path/to/internal_dataset.json \
    --external_json /path/to/external_dataset.json \
    --internal_output_dir ./results_internal \
    --external_output_dir ./results_external
"""

import argparse
import json
import os
import random
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib  # kept because your original environment may depend on it
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset as dataset
from transformers import RobertaModel

from config import (
    BACKGROUND_AS_CLASS,
    BCE_WEIGHTS,
    IN_CHANNELS,
    NUM_CLASSES,
    TRAIN_CUDA,
    TRAINING_EPOCH,
)
from dataset_external import get_train_val_test_Dataloaders
from transforms import train_transform, train_transform_cuda, val_transform, val_transform_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multimodal survival model.")

    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--seed_t", type=int, required=True, help="Random seed")

    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden parameter")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout parameter")

    parser.add_argument("--lr_image", type=float, required=True, help="Learning rate for image branch")
    parser.add_argument("--lr_report", type=float, required=True, help="Learning rate for report branch")
    parser.add_argument("--lr_total", type=float, required=True, help="Learning rate for overall model")

    parser.add_argument("--clin_hidden_dim", type=int, default=256, help="Hidden size for clinical MLP")
    parser.add_argument("--clin_layers", type=int, default=2, help="Number of MLP layers for clinical branch")
    parser.add_argument("--lr_clin", type=float, default=None, help="Learning rate for clinical branch")

    parser.add_argument(
        "--model_paths",
        type=str,
        required=True,
        help="Path to trained model (.pth) used for validation/testing",
    )

    parser.add_argument(
        "--radiobert_path",
        type=str,
        required=True,
        help="Path to pretrained RadioBERT / RoBERTa model directory",
    )

    parser.add_argument(
        "--internal_json",
        type=str,
        required=True,
        help="Path to internal dataset json",
    )

    parser.add_argument(
        "--external_json",
        type=str,
        required=True,
        help="Path to external dataset json",
    )

    parser.add_argument(
        "--internal_output_dir",
        type=str,
        default="./results_internal",
        help="Directory to save internal evaluation results",
    )

    parser.add_argument(
        "--external_output_dir",
        type=str,
        default="./results_external",
        help="Directory to save external evaluation results",
    )

    return parser.parse_args()


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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
        out = F.relu(out, inplace=True)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block: nn.Module, layers: List[int], num_classes: int = 2):
        super().__init__()
        self.in_planes = 64

        # Two-channel image input
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
        x = torch.flatten(x, 1)
        return x


def resnet18_3d(num_classes: int = 512) -> ResNet3D:
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes)


class RadioLOGIC(nn.Module):
    def __init__(self, radiobert_path: str):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(radiobert_path, add_pooling_layer=False)

    def forward(self, input_id: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_id, attention_mask=attention_mask, return_dict=False)
        sequence_output = outputs[0]
        cls_token = sequence_output[:, 0]
        return cls_token


class ImageNetBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18_3d()
        self.projection = nn.Linear(512, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.resnet(x)
        return self.projection(features)


class ReportNet(nn.Module):
    def __init__(self, radiobert_path: str):
        super().__init__()
        self.radiologic_encoder = RadioLOGIC(radiobert_path)

        for param in self.radiologic_encoder.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_id: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.radiologic_encoder(input_id, attention_mask)
        outputs = self.projection(outputs)
        return outputs


class ClinicalNet(nn.Module):
    def __init__(self, dropout_prob: float = 0.2):
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
        x = x.view(x.size(0), -1)
        return self.fc(x)


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
        fused_ir_tokens = self.fusion_stage1(tokens_stage1)
        fused_ir = fused_ir_tokens.mean(dim=1)

        tokens_stage2 = torch.stack([fused_ir, clinical_features], dim=1)
        fused_irc_tokens = self.fusion_stage2(tokens_stage2)
        fused_final = fused_irc_tokens.mean(dim=1)

        output = self.classifier(fused_final)
        return output


def cox_loss(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12, use_float64: bool = False) -> torch.Tensor:
    """
    Efron-approximated Cox partial likelihood.
    y_true: [N,2] -> time, event(1/0)
    y_pred: [N] or [N,1] risk scores (higher = higher risk)
    """
    score = y_pred.view(-1)
    time = y_true[:, 0]
    event = y_true[:, 1].bool()

    if use_float64:
        time = time.double()
        score = score.double()

    order = torch.argsort(time, descending=True)
    time = time[order]
    event = event[order]
    score = score[order]

    exp_score = torch.exp(score)
    cumsum_exp = torch.cumsum(exp_score, dim=0)

    event_mask = event
    if event_mask.sum() == 0:
        return score.sum() * 0.0

    unique_times, inv_idx = torch.unique_consecutive(time[event_mask], return_inverse=True)

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
    group_ids = inv_idx

    for g in range(len(unique_times)):
        g_mask = group_ids == g
        if not torch.any(g_mask):
            continue

        D_t = event_indices[g_mask]
        m = D_t.numel()
        num_events += m

        block_last_idx = idx_map[int(g)]
        risk_sum = cumsum_exp[block_last_idx]

        tied_score_sum = torch.sum(score[D_t])
        tied_exp_sum = torch.sum(exp_score[D_t])

        frac = torch.arange(m, dtype=score.dtype, device=score.device) / max(m, 1)
        denom_terms = torch.clamp(risk_sum - frac * tied_exp_sum, min=eps)
        sum_log_denoms = torch.sum(torch.log(denom_terms))

        total_loglik = total_loglik + tied_score_sum - sum_log_denoms

    if num_events == 0:
        return score.sum() * 0.0

    loss = -total_loglik / float(num_events)
    loss = loss + 0.0 * score.sum()
    return loss


def concordance_index_(y_true: torch.Tensor, y_pred: torch.Tensor, mode: str = "survival", tie_strategy: str = "half") -> torch.Tensor:
    """
    y_true: [N,2] -> [:,0]=time, [:,1]=event(1/0)
    y_pred: [N]   -> predicted score
        mode="survival": larger means longer survival
        mode="risk": larger means higher risk
    """
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

    if mode == "survival":
        concordant = p_i < p_j
    else:
        concordant = p_i > p_j

    score = concordant.float()
    if tie_strategy == "half":
        score = score + 0.5 * (p_i == p_j).float()

    c_index = (score * comparable.float()).sum() / denom
    return c_index


def concordance_index_test(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    actual_durations = y_true[:, 0].cpu().numpy()
    predicted_scores = y_pred.detach().cpu().numpy()
    event_observed = y_true[:, 1].cpu().numpy()

    if np.isnan(predicted_scores).any():
        return 0.5

    c_index = concordance_index(
        event_times=actual_durations,
        predicted_scores=predicted_scores,
        event_observed=event_observed,
    )
    return c_index


def cal_ci_test(y: torch.Tensor, pred: torch.Tensor, device: torch.device) -> Dict[int, float]:
    ci_dict = {}
    for target_class in range(int(y.shape[1] / 2)):
        ci_dict[target_class] = concordance_index_test(
            y[:, target_class * 2:(target_class + 1) * 2].to(device),
            -pred.to(device),
        )
    return ci_dict


def cal_loss(y: torch.Tensor, pred: torch.Tensor, criterion, device: torch.device) -> Dict[int, torch.Tensor]:
    loss_dict = {}
    for target_class in range(int(y.shape[1] / 2)):
        loss_dict[target_class] = criterion(
            y[:, target_class * 2:(target_class + 1) * 2].to(device),
            pred.to(device),
        )
    return loss_dict


def cal_ci(y: torch.Tensor, pred: torch.Tensor, device: torch.device) -> Dict[int, torch.Tensor]:
    ci_dict = {}
    for target_class in range(int(y.shape[1] / 2)):
        ci_dict[target_class] = concordance_index_(
            y[:, target_class * 2:(target_class + 1) * 2].to(device),
            -pred.to(device),
        )
    return ci_dict


def bootstrap_c_index(y_data: torch.Tensor, predictions: torch.Tensor, device: torch.device, n_iterations: int = 1000) -> Tuple[float, float, float]:
    c_indices = []
    y_np_idx = np.arange(len(y_data))

    for _ in range(n_iterations):
        indices = resample(y_np_idx)
        sample_y = y_data[indices]
        sample_preds = predictions[indices]
        c_index = cal_ci(sample_y, sample_preds, device=device)
        c_indices.append(c_index[0].detach().cpu().item())

    lower_bound = np.percentile(c_indices, 2.5)
    upper_bound = np.percentile(c_indices, 97.5)

    return lower_bound, upper_bound, float(np.mean(c_indices))


def extract_binary_horizon_labels(label_cls: torch.Tensor) -> Tuple[List[int], List[int], List[int]]:
    years_3 = []
    years_5 = []
    years_10 = []

    for tf, l in label_cls:
        tf_val = tf.detach().cpu().item()
        l_val = l.detach().cpu().item()

        if tf_val > 36:
            y3 = 0
        elif tf_val <= 36 and l_val == 1:
            y3 = 1
        else:
            y3 = -1
        years_3.append(y3)

        if tf_val > 60:
            y5 = 0
        elif tf_val <= 60 and l_val == 1:
            y5 = 1
        else:
            y5 = -1
        years_5.append(y5)

        if tf_val > 120:
            y10 = 0
        elif tf_val <= 120 and l_val == 1:
            y10 = 1
        else:
            y10 = -1
        years_10.append(y10)

    return years_3, years_5, years_10


def load_json_dataset(json_path: str) -> Dict[str, dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    training_data = dataset_json["training"]
    return {entry["identifier"]: entry for entry in training_data}


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def format_label(hr: float, ci_l: float, ci_u: float, p: float) -> str:
    if p < 0.0001:
        return f"High risk, HR = {hr:.2f} (95% CI {ci_l:.2f}–{ci_u:.2f}), p<0.0001"
    return f"High risk, HR = {hr:.2f} (95% CI {ci_l:.2f}–{ci_u:.2f}), p={p:.4f}"


def get_risk_and_new_censored(event_table: pd.DataFrame, time_points: np.ndarray) -> Tuple[List[int], List[int]]:
    at_risk = []
    censored = []

    for time in time_points:
        valid_times = event_table.index[event_table.index >= time]
        if len(valid_times) > 0:
            closest_time = valid_times.min()
            at_risk.append(int(event_table.loc[closest_time, "at_risk"]))
            censored.append(int(event_table.loc[:closest_time, "censored"].sum()))
        else:
            at_risk.append(0)
            censored.append(0)

    return at_risk, censored


def collect_predictions(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[torch.Tensor], List[Tuple[str, torch.Tensor]], List[Tuple[int, int, int, torch.Tensor]]]:
    pred_all = None
    y_all = None

    all_identifiers = []
    pres = []
    data_pre = []
    data_roc = []

    with torch.no_grad():
        for data in dl:
            image1 = data["image1"].to(device)
            image2 = data["image2"].to(device)
            label_cls = data["label_cls"].to(device)
            identifier = data["identifier"]
            report_code = data["report_code"]
            clin_features = data["clinical_features"].to(device)

            mask = report_code["attention_mask"][:, 0, :].long().to(device)
            input_id = report_code["input_ids"][:, 0, :].long().to(device)

            pred = model(image1, image2, input_id, mask, clin_features)
            y_batch = label_cls

            horizon_y3, horizon_y5, horizon_y10 = extract_binary_horizon_labels(label_cls)

            for sample_id, sample_pred in zip(identifier, pred):
                pres.append(sample_pred)
                data_pre.append((sample_id, sample_pred))
                all_identifiers.append(sample_id)

            for y3, y5, y10, sample_pred in zip(horizon_y3, horizon_y5, horizon_y10, pred):
                data_roc.append((y3, y5, y10, sample_pred))

            if pred_all is None:
                pred_all = pred
                y_all = y_batch
            else:
                pred_all = torch.cat([pred_all, pred], dim=0)
                y_all = torch.cat([y_all, y_batch], dim=0)

    return pred_all, y_all, all_identifiers, pres, data_pre, data_roc


def save_roc_table(data_roc: List[Tuple[int, int, int, torch.Tensor]], save_path: str) -> None:
    y_3, y_5, y_10, roc_scores = [], [], [], []

    for l3, l5, l10, score in data_roc:
        score_value = score.detach().cpu().numpy()[0]
        y_3.append(l3)
        y_5.append(l5)
        y_10.append(l10)
        roc_scores.append(score_value)

    df = pd.DataFrame(
        {
            "y_3": y_3,
            "y_5": y_5,
            "y_10": y_10,
            "roc_scores": roc_scores,
        }
    )
    df.to_excel(save_path, index=False)


def build_survival_dataframe(
    data_pre: List[Tuple[str, torch.Tensor]],
    pres: List[torch.Tensor],
    json_dict: Dict[str, dict],
    use_fixed_median: Optional[torch.Tensor] = None,
    include_scores: bool = False,
) -> pd.DataFrame:
    sorted_data = sorted(data_pre, key=lambda x: x[1].detach().cpu().item())

    pre_median = use_fixed_median if use_fixed_median is not None else median(pres)
    print(f"risk score median: {pre_median}")

    high_risk_ids = []
    low_risk_ids = []
    high_risk_scores = []
    low_risk_scores = []

    valid_ids = set(json_dict.keys())

    for sample_id, score in sorted_data:
        if sample_id not in valid_ids:
            continue

        if score >= pre_median:
            high_risk_ids.append(sample_id)
            high_risk_scores.append(score.detach().cpu().numpy())
        else:
            low_risk_ids.append(sample_id)
            low_risk_scores.append(score.detach().cpu().numpy())

    all_ids = low_risk_ids + high_risk_ids

    data = {
        "ID": all_ids,
        "duration": [json_dict[sid]["time"] for sid in all_ids],
        "event": [1 if json_dict[sid]["label"] == 1 else 0 for sid in all_ids],
        "primary": [1 if json_dict[sid]["primary_therapy"] == "NA" else 0 for sid in all_ids],
        "group": ["Group 1"] * len(low_risk_ids) + ["Group 2"] * len(high_risk_ids),
    }

    if include_scores:
        data["scores"] = low_risk_scores + high_risk_scores

    return pd.DataFrame(data)


def fit_cox_and_label(data: pd.DataFrame) -> Tuple[CoxPHFitter, str]:
    data_lh = data.copy()
    data_lh["group"] = data_lh["group"].astype("category").cat.codes

    cph = CoxPHFitter()
    cph.fit(data_lh, duration_col="duration", event_col="event", formula="group")

    summary = cph.summary
    p_values = summary["p"]
    hr_values = summary["exp(coef)"]
    ci_lower = summary["exp(coef) lower 95%"]
    ci_upper = summary["exp(coef) upper 95%"]

    label_highrisk = format_label(
        hr_values.iloc[0],
        ci_lower.iloc[0],
        ci_upper.iloc[0],
        p_values.iloc[0],
    )

    return cph, label_highrisk


def plot_km_curve(
    data: pd.DataFrame,
    label_highrisk: str,
    save_path: str,
) -> None:
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    mask1 = data["group"] == "Group 1"
    mask2 = data["group"] == "Group 2"

    kmf_low.fit(data["duration"][mask1], data["event"][mask1], label="Low risk: reference")
    kmf_high.fit(data["duration"][mask2], data["event"][mask2], label=label_highrisk)

    time_to_event_group1 = data["duration"][mask1]
    event_status_group1 = data["event"][mask1]
    time_to_event_group2 = data["duration"][mask2]
    event_status_group2 = data["event"][mask2]

    fig, ax = plt.subplots(figsize=(9, 7))
    plt.subplots_adjust(bottom=0.35)

    censor_styles = {"marker": "|", "ms": 5, "mew": 0.25}
    kmf_low.plot(ax=ax, ci_show=False, color="blue", show_censors=True, censor_styles=censor_styles)
    kmf_high.plot(ax=ax, ci_show=False, color="red", show_censors=True, censor_styles=censor_styles)

    time_points = np.arange(0, 121, 12)
    risk_numbers_group1, censored_numbers_group1 = get_risk_and_new_censored(kmf_low.event_table, time_points)
    risk_numbers_group2, censored_numbers_group2 = get_risk_and_new_censored(kmf_high.event_table, time_points)

    for i, t in enumerate(time_points):
        if i == 0:
            ax.add_patch(plt.Rectangle((t - 10, -0.25), 4, 0.01, color="blue", transform=ax.transData, clip_on=False))
            ax.add_patch(plt.Rectangle((t - 10, -0.35), 4, 0.01, color="red", transform=ax.transData, clip_on=False))
            ax.text(
                t - 10,
                -0.15,
                "Number at risk\n(number censored)",
                ha="left",
                va="center",
                color="black",
                fontweight="bold",
            )

        ax.text(t, -0.25, f"{risk_numbers_group1[i]}\n ({censored_numbers_group1[i]})", ha="center", va="center", color="blue")
        ax.text(t, -0.35, f"{risk_numbers_group2[i]}\n ({censored_numbers_group2[i]})", ha="center", va="center", color="red")

    time_point_5y = 60
    survival_rate_low_5y = kmf_low.predict(time_point_5y)
    survival_rate_high_5y = kmf_high.predict(time_point_5y)

    ax.axvline(x=time_point_5y, color="gray", linestyle="--")
    ax.text(time_point_5y, survival_rate_low_5y + 0.01, f"{survival_rate_low_5y:.2%}", color="blue", ha="left")
    ax.text(time_point_5y, survival_rate_high_5y - 0.05, f"{survival_rate_high_5y:.2%}", color="red", ha="right")

    print(f"5-year survival rate for Low risk group: {survival_rate_low_5y:.2%}")
    print(f"5-year survival rate for High risk group: {survival_rate_high_5y:.2%}")

    time_point_10y = 120
    survival_rate_low_10y = kmf_low.predict(time_point_10y)
    survival_rate_high_10y = kmf_high.predict(time_point_10y)

    ax.axvline(x=time_point_10y, color="gray", linestyle="--")
    ax.text(time_point_10y, survival_rate_low_10y + 0.01, f"{survival_rate_low_10y:.2%}", color="blue", ha="left")
    ax.text(time_point_10y, survival_rate_high_10y - 0.05, f"{survival_rate_high_10y:.2%}", color="red", ha="right")

    print(f"10-year survival rate for Low risk group: {survival_rate_low_10y:.2%}")
    print(f"10-year survival rate for High risk group: {survival_rate_high_10y:.2%}")

    result = logrank_test(
        time_to_event_group1,
        time_to_event_group2,
        event_observed_A=event_status_group1,
        event_observed_B=event_status_group2,
    )
    p_value = result.p_value
    print(f"P-value between Group 1 and Group 2: {p_value:.10f}")

    if p_value < 0.0001:
        label_logrank = "Log-rank test p<0.0001"
    else:
        label_logrank = f"Log-rank test p={p_value:.4f}"

    plt.plot(50, 0.6, "o", label=label_logrank, markersize=0, color="white")

    plt.title("Kaplan-Meier Survival Curve")
    plt.legend(loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Years")
    plt.ylabel("Survival Probability")

    ax.set_ylim([0.0, 1.02])
    ax.set_xlim([0.0, 120.1])
    ax.set_yticks(np.arange(0, 1.02, step=0.25))
    ax.set_xticks(np.arange(0, 120.1, step=12))
    ax.set_xticklabels([f"{int(x / 12)}" for x in np.arange(0, 120.1, step=12)])

    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)


def plot_cox_summary(cph: CoxPHFitter, save_path: str) -> None:
    plt.figure(figsize=(5, 5))
    ax = cph.plot(hazard_ratios=False, c="black", marker="o", markersize=8, markerfacecolor="black")
    ax.set_xlim([-1.0, 5.1])
    ax.set_xticks(np.arange(-1.0, 5.1, step=0.5))
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.close()


def evaluate_single_dataloader(
    model: nn.Module,
    dl: DataLoader,
    dataloader_name: str,
    json_path: str,
    output_dir: str,
    device: torch.device,
    hidden_dim_value: int,
    dropout_value: float,
    cycles_date: str,
    fixed_median: Optional[torch.Tensor] = None,
    include_scores: bool = False,
) -> None:
    ensure_dir(output_dir)

    print("------------------------------------------------")
    print("------------------------------------------------")
    print(f"------ {dataloader_name} ------ {dataloader_name} ------ {dataloader_name} ------")
    print("------------------------------------------------")
    print("------------------------------------------------")

    pred_all, y_all, _, pres, data_pre, data_roc = collect_predictions(model, dl, device=device)

    ci_dict = cal_ci_test(y_all, pred_all, device=device)

    print("------------------------------------")
    print("hidden_dim_value:", hidden_dim_value)
    print("dropout_value:", dropout_value)
    print("------------------------------------")
    print("C-index 95%  ---")
    print(ci_dict)
    print(bootstrap_c_index(y_all, pred_all, device=device))

    json_dict = load_json_dataset(json_path)

    survival_df = build_survival_dataframe(
        data_pre=data_pre,
        pres=pres,
        json_dict=json_dict,
        use_fixed_median=fixed_median,
        include_scores=include_scores,
    )

    primary_set = "All"
    data_set = dataloader_name

    roc_excel = os.path.join(output_dir, f"{cycles_date}_{primary_set}_{data_set}_ROC.xlsx")
    cph_excel = os.path.join(output_dir, f"{cycles_date}_{primary_set}_{data_set}_cph_image_report.xlsx")
    km_png = os.path.join(output_dir, f"{cycles_date}_{primary_set}_{data_set}_Kaplan-Meier_image_report.png")
    cox_png = os.path.join(output_dir, f"{cycles_date}_{primary_set}_{data_set}_cph_image_report.png")

    save_roc_table(data_roc, roc_excel)

    cph, label_highrisk = fit_cox_and_label(survival_df)
    survival_df.to_excel(cph_excel, index=False)

    plot_km_curve(survival_df.copy(), label_highrisk=label_highrisk, save_path=km_png)

    cph_ready_df = survival_df.copy()
    cph_ready_df["group"] = cph_ready_df["group"].astype("category").cat.codes

    cph_final = CoxPHFitter()
    cph_final.fit(cph_ready_df, duration_col="duration", event_col="event", formula="group")
    cph_final.print_summary()
    print("-----------------------")
    print("Log-rank test, p_value:", cph_final.summary["p"])
    print("-----------------------")

    plot_cox_summary(cph_final, cox_png)


def compute_validation_median(
    model: nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        _, _, _, pres, _, _ = collect_predictions(model, val_dataloader, device=device)
    pre_median_train = median(pres)
    print("pre_median_train:", pre_median_train)
    return pre_median_train


def main() -> None:
    args = parse_args()
    setup_seed(args.seed_t)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"seed: {args.seed_t}  model: {args.model_paths}")

    _num_classes = NUM_CLASSES + 1 if BACKGROUND_AS_CLASS else NUM_CLASSES

    model = MultiModalModel(
        radiobert_path=args.radiobert_path,
        dropout_prob=args.dropout,
    ).to(device)

    state_dict = torch.load(args.model_paths, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    train_dataloader, val_dataloader, test_dataloader, test_dataloader_external = get_train_val_test_Dataloaders()

    # Use validation median as internal threshold, consistent with your original logic
    pre_median_train = compute_validation_median(model, val_dataloader, device=device)

    internal_dataloaders = {
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
    }

    for dataloader_name, dl in internal_dataloaders.items():
        evaluate_single_dataloader(
            model=model,
            dl=dl,
            dataloader_name=dataloader_name,
            json_path=args.internal_json,
            output_dir=args.internal_output_dir,
            device=device,
            hidden_dim_value=args.hidden_dim,
            dropout_value=args.dropout,
            cycles_date=args.name,
            fixed_median=pre_median_train,
            include_scores=False,
        )

    external_dataloaders = {
        "test_dataloader_external": test_dataloader_external,
    }

    for dataloader_name, dl in external_dataloaders.items():
        evaluate_single_dataloader(
            model=model,
            dl=dl,
            dataloader_name=dataloader_name,
            json_path=args.external_json,
            output_dir=args.external_output_dir,
            device=device,
            hidden_dim_value=args.hidden_dim,
            dropout_value=args.dropout,
            cycles_date=args.name,
            fixed_median=None,
            include_scores=True,
        )


if __name__ == "__main__":
    main()