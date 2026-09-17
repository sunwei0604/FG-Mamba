import os
import numpy as np
import random
from scipy import io
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset


DSFORMER_SEEDS = (
    202401,
    202402,
    202403,
    202404,
    202405,
    202406,
    202407,
    10001,
    202409,
    202410,
)


DATASET_CONFIG = {
    'ip': {
        'folder': 'Indian pine',
        'img_file': 'Indian_pines_corrected.mat', 'img_key': 'indian_pines_corrected',
        'gt_file': 'Indian_pines_gt.mat', 'gt_key': 'indian_pines_gt',
        'train_samples_per_class': 50,
    },
    'pu': {
        'folder': 'PU',
        'img_file': 'PaviaU.mat', 'img_key': 'paviaU',
        'gt_file': 'PaviaU_gt.mat', 'gt_key': 'paviaU_gt',
        'train_samples_per_class': 30,
    },
    'salinas': {
        'folder': 'Salinas',
        'img_file': 'Salinas_corrected.mat', 'img_key': 'salinas_corrected',
        'gt_file': 'Salinas_gt.mat', 'gt_key': 'salinas_gt',
        'train_samples_per_class': 50,
    },
    'houston': {
        'folder': 'Houston',
        'img_file': 'HSI.mat', 'img_key': 'HSI',
        'gt_file': 'gt.mat', 'gt_key': 'gt',
        'train_samples_per_class': 50,
    },
    'whu_hanchuan': {
        'folder': 'WHU-Hi-HanChuan',
        'img_file': 'WHU_Hi_HanChuan.mat', 'img_key': 'WHU_Hi_HanChuan',
        'gt_file': 'WHU_Hi_HanChuan_gt.mat', 'gt_key': 'WHU_Hi_HanChuan_gt',
        'train_samples_per_class': 50,
    },
    'whu_honghu': {
        'folder': 'WHU-Hi-HongHu',
        'img_file': 'WHU_Hi_HongHu.mat', 'img_key': 'WHU_Hi_HongHu',
        'gt_file': 'WHU_Hi_HongHu_gt.mat', 'gt_key': 'WHU_Hi_HongHu_gt',
        'train_samples_per_class': 50,
    },
    'whu_longkou': {
        'folder': 'WHU-Hi-LongKou',
        'img_file': 'WHU_Hi_LongKou.mat', 'img_key': 'WHU_Hi_LongKou',
        'gt_file': 'WHU_Hi_LongKou_gt.mat', 'gt_key': 'WHU_Hi_LongKou_gt',
        'train_samples_per_class': 50,
    },
}

DATASET_ALIASES = {
    'houston13': 'houston',
    'whuhh': 'whu_honghu',
}

SUPPORTED_DATASETS = tuple(DATASET_CONFIG.keys()) + tuple(DATASET_ALIASES.keys())

DATASET_CLASS_NAMES = {
    'pu': [
        'Asphalt', 'Meadows', 'Gravel', 'Trees',
        'Painted metal sheets', 'Bare Soil', 'Bitumen',
        'Self-Blocking Bricks', 'Shadows'
    ],
    'ip': [
        'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
        'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
        'Soybean-clean', 'Wheat', 'Woods',
        'Buildings-Grass-Trees-Drives', 'Stone-Steel-Towers'
    ],
    'salinas': [
        'Broccoli green weeds 1', 'Broccoli green weeds 2',
        'Fallow', 'Fallow rough plow', 'Fallow smooth',
        'Stubble', 'Celery', 'Grapes untrained',
        'Soil vineyard develop', 'Corn senesced green weeds',
        'Lettuce romaine 4wk', 'Lettuce romaine 5wk',
        'Lettuce romaine 6wk', 'Lettuce romaine 7wk',
        'Vineyard untrained', 'Vineyard vertical trellis'
    ],
    'houston': [
        'Healthy grass', 'Stressed grass', 'Synthetic grass',
        'Trees', 'Soil', 'Water', 'Residential',
        'Commercial', 'Road', 'Highway', 'Railway',
        'Parking lot 1', 'Parking lot 2', 'Tennis court', 'Running track'
    ],
    'whu_honghu': [
        'Red roof', 'Road', 'Bare soil', 'Cotton',
        'Cotton firewood', 'Rape', 'Chinese cabbage', 'Pakchoi',
        'Cabbage', 'Tuber mustard', 'Brassica parachinensis',
        'Brassica chinensis', 'Small Brassica chinensis', 'Lactuca sativa',
        'Celtuce', 'Film covered lettuce', 'Romaine lettuce',
        'Carrot', 'White radish', 'Garlic sprout',
        'Broad bean', 'Tree'
    ],
}

for _alias, _target in DATASET_ALIASES.items():
    if _target in DATASET_CLASS_NAMES:
        DATASET_CLASS_NAMES[_alias] = DATASET_CLASS_NAMES[_target]


def canonical_dataset_name(dataset_name):
    name = str(dataset_name).strip()
    return DATASET_ALIASES.get(name, name)


def get_default_samples_per_class(dataset_name):
    name = canonical_dataset_name(dataset_name)
    if name not in DATASET_CONFIG:
        raise ValueError(f"Dataset {dataset_name} not found in config. Available: {list(SUPPORTED_DATASETS)}")
    return int(DATASET_CONFIG[name].get('train_samples_per_class', 50))


def _find_dataset_file(dataset_dir, cfg, file_name):
    candidates = [
        os.path.join(dataset_dir, cfg['folder'], file_name),
        os.path.join(dataset_dir, file_name),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    matches = []
    for root, _, files in os.walk(dataset_dir):
        if file_name in files:
            matches.append(os.path.join(root, file_name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        expected = os.path.join(dataset_dir, cfg['folder'], file_name)
        matches_text = "\n    ".join(matches)
        raise FileNotFoundError(
            f"Found multiple '{file_name}' under '{dataset_dir}'. "
            f"Please keep the expected file at '{expected}' or remove duplicates:\n    {matches_text}"
        )
    expected = os.path.join(dataset_dir, cfg['folder'], file_name)
    raise FileNotFoundError(
        f"Cannot find '{file_name}'. Expected: {expected}. "
        f"Dataset root currently used: {os.path.abspath(dataset_dir)}"
    )


def _load_mat_var(path, key):
    payload = io.loadmat(path)
    if key not in payload:
        keys = [k for k in payload.keys() if not k.startswith('__')]
        raise KeyError(f"Key '{key}' not found in {path}. Available keys: {keys}")
    return payload[key]


def load_hsi_data(dataset_name, dataset_dir):
    dataset_name = canonical_dataset_name(dataset_name)
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset {dataset_name} not found in config. Available: {list(SUPPORTED_DATASETS)}")
    cfg = DATASET_CONFIG[dataset_name]
    img_path = _find_dataset_file(dataset_dir, cfg, cfg['img_file'])
    gt_path = _find_dataset_file(dataset_dir, cfg, cfg['gt_file'])
    print(f"    Data file : {img_path}")
    print(f"    GT file   : {gt_path}")
    image = _load_mat_var(img_path, cfg['img_key'])
    gt = _load_mat_var(gt_path, cfg['gt_key'])
    nan_mask = np.isnan(image.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        image[nan_mask] = 0
        gt[nan_mask] = 0
    image = np.asarray(image, dtype=np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    mean_by_c = np.mean(image, axis=(0, 1))
    image = image - mean_by_c
    gt = gt.astype(int) - 1
    return image, gt


def apply_pca(image, n_components=1):
    H, W, C = image.shape
    reshaped_image = image.reshape(-1, C)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped_image)
    return pca_result.reshape(H, W, n_components)


def sample_train_test_dsformer(gt, samples_per_class, seed):
    np.random.seed(seed)
    random.seed(seed)
    class_indices = {}
    for idx in zip(*np.where(gt >= 0)):
        label = int(gt[idx])
        class_indices.setdefault(label, []).append(idx)
    train_gt = np.full_like(gt, -1)
    test_gt = np.full_like(gt, -1)
    requested = int(samples_per_class)
    for label, idx_list in class_indices.items():
        n_train = 15 if len(idx_list) < requested else requested
        if len(idx_list) < n_train:
            raise ValueError(
                f"DSFormer 至少需要 15 个样本，但类别 {label} 只有 {len(idx_list)} 个。"
            )
        train_sel = set(random.sample(idx_list, n_train))
        for idx in idx_list:
            if idx in train_sel:
                train_gt[idx] = label
            else:
                test_gt[idx] = label
    return train_gt, test_gt


class HSIPatchDataset(Dataset):

    def __init__(self, raw_image, gt, patch_size, pca_image=None, data_aug=False):
        super().__init__()
        self.patch_size = patch_size
        self.data_aug = data_aug
        self.gt = gt
        self.indices = np.argwhere(gt >= 0)
        ps = patch_size // 2
        self.pad_raw = np.pad(raw_image, ((ps, ps), (ps, ps), (0, 0)), mode='reflect')
        self.pad_pca = None
        if pca_image is not None:
            self.pad_pca = np.pad(pca_image, ((ps, ps), (ps, ps), (0, 0)), mode='reflect')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        r, c = self.indices[idx]
        patch_raw = self.pad_raw[r: r + self.patch_size, c: c + self.patch_size, :]
        if self.pad_pca is not None:
            patch_pca = self.pad_pca[r: r + self.patch_size, c: c + self.patch_size, :]
        else:
            patch_pca = np.zeros((self.patch_size, self.patch_size, 1))
        if self.data_aug:
            patch_raw = patch_raw.copy()
            patch_pca = patch_pca.copy()
            flip_type = np.random.randint(0, 3)
            if flip_type == 0:
                patch_raw = np.fliplr(patch_raw)
                patch_pca = np.fliplr(patch_pca)
            elif flip_type == 1:
                patch_raw = np.flipud(patch_raw)
                patch_pca = np.flipud(patch_pca)
            rot_times = np.random.randint(0, 4)
            patch_raw = np.rot90(patch_raw, rot_times)
            patch_pca = np.rot90(patch_pca, rot_times)
        patch_raw = torch.tensor(patch_raw.copy().transpose(2, 0, 1), dtype=torch.float32)
        patch_pca = torch.tensor(patch_pca.copy().transpose(2, 0, 1), dtype=torch.float32)
        target = torch.tensor(self.gt[r, c], dtype=torch.long)
        return patch_raw, patch_pca, target
