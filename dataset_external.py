import copy
import json
import os
import random
import tarfile
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from transformers import RobertaTokenizerFast

from config_new import (
    DATASET_PATH,
    TASK_ID,
    TRAIN_VAL_TEST_SPLIT,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    TEST_BATCH_SIZE,
    SEED,
)

# Optional config values in config_new.py
try:
    from config_new import META_PATH
except ImportError:
    META_PATH = os.environ.get("META_PATH", "./dataset.json")

try:
    from config_new import EXTERNAL_JSON_PATH
except ImportError:
    EXTERNAL_JSON_PATH = os.environ.get("EXTERNAL_JSON_PATH", "./external")

try:
    from config_new import TOKENIZER_PATH
except ImportError:
    TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "roberta-base")

try:
    from config_new import NUM_WORKERS
except ImportError:
    NUM_WORKERS = 4

try:
    from config_new import PIN_MEMORY
except ImportError:
    PIN_MEMORY = False


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


setup_seed(SEED)

_TOKENIZER = None


def get_tokenizer():
    """
    Lazy-load tokenizer to avoid loading model files at import time.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = RobertaTokenizerFast.from_pretrained(
            TOKENIZER_PATH,
            return_tensors="pt",
        )
    return _TOKENIZER


def extract_tar(tar_path: str, output_dir: str) -> None:
    """
    Extract a .tar file into the target directory.
    """
    try:
        print("Extracting tar file...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(output_dir)
    except Exception as exc:
        raise RuntimeError("File extraction failed.") from exc
    print("Extraction completed!")


def clip_and_norm_old(arr, vmin=0, vmax=3000):
    arr = np.clip(arr, vmin, vmax).astype(np.float32)
    maxv = arr.max()
    if maxv > 0:
        arr = arr / maxv
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def clip_and_norm(arr, vmin=0, vmax=3000):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, vmin, vmax)
    max_val = arr.max()
    min_val = arr.min()
    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    return np.zeros_like(arr, dtype=np.float32)


def normalize_dce_pair_all(dce1: np.ndarray, dce2: np.ndarray):
    """
    Normalize two DCE volumes using the 99th percentile of dce2 as the shared upper bound.
    """
    max_val = np.percentile(dce2, 99)
    if max_val < 1e-6:
        max_val = 1e-6
    dce1 = np.clip(dce1, 0, max_val) / max_val
    dce2 = np.clip(dce2, 0, max_val) / max_val
    return dce1.astype(np.float32), dce2.astype(np.float32)


task_names = {
    "01": "a_br_survival_data_2024",
    "02": "Heart",
    "03": "Liver",
    "04": "Hippocampus",
    "05": "Prostate",
    "06": "Lung",
    "07": "Pancreas",
    "08": "HepaticVessel",
    "09": "Spleen",
    "10": "Colon",
    "11": "a_br_survival_data",
}


THERAPY_KEYS = [
    "Neoadjuvant Radiation Therapy",
    "Adjuvant Radiation Therapy",
    "Neoadjuvant Chemotherapy",
    "Adjuvant Chemotherapy",
    "Neoadjuvant Endocrine Therapy Medications",
    "Adjuvant Endocrine Therapy Medications",
    "Neoadjuvant Anti-Her2 Neu Therapy",
    "Adjuvant Anti-Her2 Neu Therapy",
]


def encode_therapies(item: dict) -> np.ndarray:
    """
    Build a 10-dimensional therapy encoding:
    - 8 dims for therapy keys (>0 -> 1, else 0)
    - 1 dim for whether primary_therapy is 'Surgery'
    - 1 dim for surgery flag (field name kept as 'surgey' to match source data)
    """
    vec = []
    for key in THERAPY_KEYS:
        value = item.get(key, 0)
        try:
            value = float(value)
        except Exception:
            value = 0.0
        vec.append(1.0 if value > 0 else 0.0)

    primary = item.get("primary_therapy", "NA")
    vec.append(
        1.0 if isinstance(primary, str) and primary.strip().lower() == "surgery" else 0.0
    )

    surgery_flag = item.get("surgey", 0)
    try:
        surgery_flag = int(surgery_flag)
    except Exception:
        surgery_flag = 0
    vec.append(1.0 if surgery_flag == 1 else 0.0)

    return np.asarray(vec, dtype=np.float32)


def generate_one_hot(
    T_stage_value,
    N_stage_value,
    # M_stage_value,
    T_stage_post_value,
    N_stage_post_value,
    # M_stage_post_value,
    family_history,
    age,
    E,
    P,
    H,
    tumor_types,
):
    one_hot_matrix = np.zeros((25, 12), dtype=np.float32)

    T_stage = ['0', '1', '1A', '1B', '1C', '1M', '1MI', '2', '3', '4', '4A', '4B', '4D', 'IS', 'X']
    N_stage = ['0', '0IS', '0S', '1', '1MS', '1S', '1BS', '2A', '2B', '3', '3A', '3B', '3BS', '3C', 'X']
    # M_stage = ['0', '1', 'X']
    T_stage_post = ['0', '1', '1A', '1B', '1C', '1MI', '2', '3', '4B', 'IS', 'X', 'Y0', 'Y1', 'Y1A', 'Y1B', 'Y1C', 'Y1MI', 'Y2', 'Y3', 'Y4A', 'Y4B', 'Y4D', 'YIS', 'YX']
    N_stage_post = ['0', '0I', '0IS', '0S', '1', '1A', '1AS', '1B', '1BS', '1B1', '1B2', '1B3', '1B4', '1M', '1MI', '1MS', '2', '2A', '2B', '3A', '3B', '3C', 'X']
    # M_stage_post = ['0', '1', 'X']

    def set_hot(value, vocab, col):
        if value not in ["-1", "-", "None", None]:
            value = str(value).strip()
            if value in vocab:
                one_hot_matrix[vocab.index(value), col] = 1.0

    set_hot(T_stage_value, T_stage, 0)
    set_hot(N_stage_value, N_stage, 1)
    # set_hot(M_stage_value, M_stage, 2)
    set_hot(T_stage_post_value, T_stage_post, 3)
    set_hot(N_stage_post_value, N_stage_post, 4)
    # set_hot(M_stage_post_value, M_stage_post, 5)

    if isinstance(family_history, str) and ('kanker' in family_history.lower()):
        one_hot_matrix[0, 6] = 1.0

    try:
        age_f = float(age)
    except Exception:
        age_f = -1.0

    if age_f >= 0:
        bucket = int(age_f // 4)
        bucket = max(0, min(24, bucket))
        one_hot_matrix[bucket, 7] = 1.0

    def bin_eph(value, col):
        try:
            value = float(value)
        except Exception:
            value = -1.0
        if value < 0:
            one_hot_matrix[:, col] = 0.0
            return
        flag = (0 < (value / 4) < 25)
        one_hot_matrix[1 if flag else 0, col] = 1.0

    bin_eph(E, 8)
    bin_eph(P, 9)
    bin_eph(H, 10)

    if isinstance(tumor_types, str):
        s = tumor_types.lower()
        if 'ductaal' in s and 'infiltrerend ductaal' not in s and 'intraductaal carcinoom' not in s and 'ductaal carcinoma in situ' not in s:
            one_hot_matrix[0, 11] = 1.0
        if 'infiltrerend ductaal' in s and 'intraductaal carcinoom' not in s:
            one_hot_matrix[1, 11] = 1.0
        if 'lobulair' in s and 'infiltrerend lobulair' not in s:
            one_hot_matrix[2, 11] = 1.0
        if 'infiltrerend lobulair' in s:
            one_hot_matrix[3, 11] = 1.0
        if 'tubular' in s:
            one_hot_matrix[4, 11] = 1.0
        if 'mucineus' in s:
            one_hot_matrix[5, 11] = 1.0
        if 'micropapillair' in s:
            one_hot_matrix[6, 11] = 1.0
        if 'papillair' in s and 'micropapillair' not in s and 'intraductaal papillair adenocarcinoom' not in s:
            one_hot_matrix[7, 11] = 1.0
        if 'ductaal carcinoma in situ' in s or 'intraductaal carcinoom' in s or 'intraductaal papillair adenocarcinoom' in s:
            one_hot_matrix[8, 11] = 1.0
        if one_hot_matrix[:, 11].sum() == 0:
            one_hot_matrix[9, 11] = 1.0

    return one_hot_matrix.reshape(-1).astype(np.float32)


def _normalize_stage_string(stage):
    if stage is None:
        return "-1"
    return str(stage).strip().upper()


def update_stage(stage):
    """
    Coarse stage mapping.
    Keeps original behavior as much as possible.
    """
    stage = _normalize_stage_string(stage)
    if stage == "-1":
        return "-1"
    if "0" in stage:
        return "0"
    if "1" in stage:
        return "1"
    if "2" in stage:
        return "2"
    if "3" in stage:
        return "3"
    if "4" in stage:
        return "4"
    if "IS" in stage:
        return "IS"
    if "X" in stage:
        return "X"
    return "-1"


def update_n_post_stage(stage):
    stage = _normalize_stage_string(stage)
    if stage == "-1":
        return "-1"
    if "0" in stage:
        return "0"
    if "1" in stage:
        return "-1"
    return "-1"


class MedicalSegmentationDecathlon(Dataset):
    def __init__(
        self,
        task_number,
        dir_path,
        split_ratios=None,
        transforms=None,
        mode=None,
        external_json_path=None,
        meta_path=None,
        tokenizer_path=None,
    ) -> None:
        super().__init__()
        self.task_number = str(task_number).zfill(2)
        self.file_name = f"Task{self.task_number}_{task_names[self.task_number]}"
        self.dir = os.path.join(dir_path)
        self.transform = transforms
        self.splits = split_ratios or [0.5, 0.25, 0.25]
        self.mode = mode
        self.external_json_path = external_json_path or EXTERNAL_JSON_PATH
        self.meta_path = meta_path or META_PATH
        self.tokenizer_path = tokenizer_path or TOKENIZER_PATH

        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

        raw = self.meta["training"]

        def _key(item):
            img = str(item.get("image", ""))
            name = img.split("/")[-1]
            return (name, str(item.get("identifier", "NA")))

        self.samples = sorted(raw, key=_key)

        self.train = None
        self.val = None
        self.test = None

        self.external_test = []
        if self.external_json_path:
            json_path = os.path.join(self.external_json_path, "duke_survival.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                self.external_test = data["training"]

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "external_test":
            return len(self.external_test)
        return len(self.samples)

    def _build_internal_paths(self, name):
        img_path1 = os.path.join(self.dir, "MRI_single_crop", name + "_1.nii.gz")
        img_path2 = os.path.join(self.dir, "MRI_single_crop", name + "_2.nii.gz")
        img_path3 = os.path.join(self.dir, "MRI_single_crop", name + "_mask.nii.gz")
        return img_path1, img_path2, img_path3

    def _build_external_paths(self, name):
        img_path1 = os.path.join(self.external_json_path, "Duke_pro", name, "dce1.nii.gz")
        img_path2 = os.path.join(self.external_json_path, "Duke_pro", name, "dce2.nii.gz")
        img_path3 = os.path.join(self.external_json_path, "Duke_pro", name, "seg.nii.gz")
        return img_path1, img_path2, img_path3

    def __getitem__(self, idx):
        item = self.external_test[idx] if self.mode == "external_test" else self.samples[idx]

        name = item["image"].split("/")[-1].replace(".nii.gz", "")

        if self.mode == "external_test":
            img_path1, img_path2, img_path3 = self._build_external_paths(name)
        else:
            img_path1, img_path2, img_path3 = self._build_internal_paths(name)

        if not (os.path.exists(img_path1) and os.path.exists(img_path2) and os.path.exists(img_path3)):
            return None

        img_array1 = nib.load(img_path1).get_fdata().astype(np.float32)
        img_array2 = nib.load(img_path2).get_fdata().astype(np.float32)
        img_array3 = nib.load(img_path3).get_fdata().astype(np.float32)

        img_array1, img_array2 = normalize_dce_pair_all(img_array1, img_array2)

        mask_modified = np.where(img_array3 == 1, 10, 1).astype(np.float32)
        mask_modified = np.expand_dims(mask_modified, axis=0)

        img_array1 = np.expand_dims(img_array1, axis=0).astype(np.float32)
        img_array2 = np.expand_dims(img_array2, axis=0).astype(np.float32)

        img_array1 = img_array1 * mask_modified
        img_array2 = img_array2 * mask_modified

        label = int(item.get("label", 0))

        time_diff = -1
        if "time" in item:
            if item["time"] is not None:
                time_diff = int(item["time"])
            else:
                time_diff = 0

        label_cls = np.array([time_diff, label], dtype=int)

        label_drm = int(item.get("label_drm", 0))
        time_drm = -1
        if "time_drm" in item:
            if item["time_drm"] is not None:
                time_drm = int(item["time_drm"])
            else:
                time_drm = 0

        label_drm_free = np.array([time_drm, label_drm], dtype=int)

        eph = item.get("EPH_surv", [-1, -1, -1])
        E, P, H = eph

        report = item.get("reports_surv", "None") or "None"
        tumor_types = item.get("tumor_types", "-1")
        T_stage = item.get("T_stage", "-1")
        N_stage = item.get("N_stage", "-1")
        # M_stage = item.get("M_stage", "-1")
        T_stage_post = item.get("T_stage_post", "-1")
        N_stage_post = item.get("N_stage_post", "-1")
        # M_stage_post = item.get("M_stage_post", "-1")
        family_history = item.get("family_history", "NA")
        age = item.get("AGE", -1)
        if age is None:
            age = -1

        T_stage = update_stage(T_stage)
        N_stage = update_stage(N_stage)
        # M_stage = update_stage(M_stage)
        T_stage_post = update_stage(T_stage_post)
        N_stage_post = update_n_post_stage(N_stage_post)
        # M_stage_post = update_stage(M_stage_post)

        report2 = report.replace("\n", "")
        index_klinische = report2.find("Klinische")
        if index_klinische != -1:
            index_verslag = report2.find("Verslag", index_klinische)
            if index_verslag != -1:
                report_clean = report2[index_verslag:]
            else:
                report_clean = report2
        else:
            report_clean = report2

        tokenizer = get_tokenizer()
        report_code = tokenizer(
            report_clean,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        base_feats = generate_one_hot(
            T_stage,
            N_stage,
            # M_stage,
            T_stage_post,
            N_stage_post,
            # M_stage_post,
            family_history,
            age,
            E,
            P,
            H,
            tumor_types,
        )

        therapy_feats = encode_therapies(item)
        clin_features = np.concatenate([base_feats, therapy_feats], axis=0).astype(np.float32)

        identifier = item.get("identifier", "NA")
        id_date = item.get("ID_PI", "NA")

        return {
            "clinical_features": clin_features,
            "primary": item.get("primary_therapy", "NA"),
            "image1": img_array1,
            "image2": img_array2,
            "label_cls": label_cls,
            "label_cls_drm": label_drm_free,
            "time_diff": time_diff,
            "identifier": identifier,
            "id_date": id_date,
            "report_code": report_code,
            "eph_code": np.array(eph, dtype=int),
        }


def skip_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def _extract_event_labels(samples):
    """
    Extract event labels for stratified splitting.
    Prefer 'label_drm', then 'label', else 0.
    """
    y = []
    for item in samples:
        if "label_drm" in item and item["label_drm"] is not None:
            y.append(int(item["label_drm"]))
        elif "label" in item and item["label"] is not None:
            y.append(int(item["label"]))
        else:
            y.append(0)
    return np.array(y, dtype=np.int64)


def get_train_val_test_Dataloaders(
    external_json_path=EXTERNAL_JSON_PATH,
    seed_split_train=2026,
    seed_split_valtest=2026,
    meta_path=META_PATH,
):
    """
    Build train/val/test dataloaders with stratified splitting.
    """
    full_dataset = MedicalSegmentationDecathlon(
        task_number=TASK_ID,
        dir_path=DATASET_PATH,
        split_ratios=TRAIN_VAL_TEST_SPLIT,
        transforms=None,
        external_json_path=external_json_path,
        meta_path=meta_path,
        tokenizer_path=TOKENIZER_PATH,
    )

    samples = full_dataset.samples
    n_samples = len(samples)
    all_idx = np.arange(n_samples)

    y_event = _extract_event_labels(samples)

    p_train, p_val, p_test = TRAIN_VAL_TEST_SPLIT
    assert abs(p_train + p_val + p_test - 1.0) < 1e-6, "TRAIN_VAL_TEST_SPLIT must sum to 1."

    n_train = int(round(p_train * n_samples))
    n_val = int(round(p_val * n_samples))
    n_test = n_samples - n_train - n_val

    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_train,
        random_state=seed_split_train,
    )
    train_indices, rest_indices = next(sss1.split(all_idx, y_event))

    y_rest = y_event[rest_indices]
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_val,
        test_size=len(rest_indices) - n_val,
        random_state=seed_split_valtest,
    )
    val_sub, test_sub = next(sss2.split(rest_indices, y_rest))
    val_indices = rest_indices[val_sub]
    test_indices = rest_indices[test_sub]

    def _make_subset(idx_list, mode_name):
        ds = copy.deepcopy(full_dataset)
        ds.samples = [samples[i] for i in idx_list]
        ds.set_mode(mode_name)
        return ds

    train_set = _make_subset(train_indices, "train")
    val_set = _make_subset(val_indices, "val")
    test_set = _make_subset(test_indices, "test")

    print(f"Stratified split complete: Train={len(train_set)}  Val={len(val_set)}  Test={len(test_set)}")

    def _rate(ds):
        y = _extract_event_labels(ds.samples)
        return float(y.mean()) if len(y) > 0 else 0.0

    print(f"Event rate: Train={_rate(train_set):.3f}  Val={_rate(val_set):.3f}  Test={_rate(test_set):.3f}")

    train_loader = DataLoader(
        train_set,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=skip_none_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=skip_none_collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=skip_none_collate,
    )

    external_test_set = copy.deepcopy(full_dataset)
    external_test_set.set_mode("external_test")
    test_loader_external = DataLoader(
        external_test_set,
        batch_size=TEST_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=PIN_MEMORY,
        collate_fn=skip_none_collate,
    )

    return train_loader, val_loader, test_loader, test_loader_external