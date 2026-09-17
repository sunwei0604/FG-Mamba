import argparse
import os
import random
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import (
    DATASET_CLASS_NAMES,
    DSFORMER_SEEDS,
    HSIPatchDataset,
    SUPPORTED_DATASETS,
    apply_pca,
    canonical_dataset_name,
    get_default_samples_per_class,
    load_hsi_data,
    sample_train_test_dsformer,
)
from utils.loss import FocalOrthogonalLoss
from utils.flops import profile_model
from models.get_model import create_model
from train import train_epoch, evaluate, inference_and_draw


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_ROOT = os.path.join(PROJECT_ROOT, "logs_and_checkpoints")


def resolve_save_dir(save_dir):
    if save_dir is None or str(save_dir).strip() == "":
        return DEFAULT_SAVE_ROOT
    raw = str(save_dir).strip().replace("\\", os.sep).replace("/", os.sep)
    raw = os.path.normpath(os.path.expanduser(raw))
    if os.path.isabs(raw):
        abs_path = os.path.abspath(raw)
        try:
            inside_default_root = os.path.commonpath([abs_path, DEFAULT_SAVE_ROOT]) == DEFAULT_SAVE_ROOT
        except ValueError:
            inside_default_root = False
        if inside_default_root:
            return abs_path
        redirected = os.path.join(DEFAULT_SAVE_ROOT, os.path.basename(abs_path.rstrip(os.sep)))
        print(f"[save_dir] Redirected '{save_dir}' -> '{redirected}'")
        return redirected
    rel = raw.strip(os.sep)
    if rel in ("", ".", "logs_and_checkpoints"):
        return DEFAULT_SAVE_ROOT
    if rel.startswith("logs_and_checkpoints" + os.sep):
        return os.path.abspath(os.path.join(PROJECT_ROOT, rel))
    redirected = os.path.join(DEFAULT_SAVE_ROOT, os.path.basename(rel))
    print(f"[save_dir] Redirected relative save_dir '{save_dir}' -> '{redirected}'")
    return redirected


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_class_weights(gt, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        counts[c] = np.sum(gt == c)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return weights.tolist()


def print_run_header(run, num_runs, seed, dataset, samples_per_class):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Run {run:>2d} / {num_runs}  |  seed={seed}"
          f"  |  dataset={dataset}  |  spc={samples_per_class}")
    print(sep)


def format_class_acc_with_std(mean_acc, std_acc, class_names):
    lines = []
    max_name_len = max(len(n) for n in class_names)
    for i, name in enumerate(class_names):
        padded = name.ljust(max_name_len)
        lines.append(f"    {padded} : {mean_acc[i]:.2f} ± {std_acc[i]:.2f} %")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="FG-Mamba Training Pipeline")
    parser.add_argument('--device', type=str, default='0', choices=['0', '1'],
                        help='GPU 设备编号（0 或 1）')
    parser.add_argument('--dataset', type=str, default='whu_honghu',
                        choices=list(SUPPORTED_DATASETS),
                        help='数据集名称')
    parser.add_argument('--dataset_dir', type=str, default='./datasets',
                        help='数据集根目录')
    parser.add_argument('--model', type=str, default='fg_mamba', choices=['fg_mamba'],
                        help='完整 FG-Mamba 模型')
    parser.add_argument('--patch_size', type=int, default=25,
                        help='输入切片大小（奇数）')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='特征嵌入维度')
    parser.add_argument('--depth', type=int, default=6,
                        help='FG_Mamba_Block 堆叠层数')
    parser.add_argument('--samples_per_class', type=int, default=None,
                        help='每类训练样本数（默认：PU=30，其余=50）')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数（文章设为 300）')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker count; server recommendation: 4 or 8')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable CUDA FP16 automatic mixed precision')
    parser.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable TF32 on supported NVIDIA GPUs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='每隔多少个 epoch 打印一次训练损失')
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--lambda_ortho', type=float, default=0.001,
                        help='正交约束损失系数（设 0 等价于 w/o ortho reg）')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='权重和日志保存目录')
    parser.add_argument('--save_splits', action='store_true', default=False,
                        help='Save DSFormer train/test split masks as .npz files. Default: false.')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='独立重复实验次数')
    parser.add_argument('--draw_map', action='store_true', default=True,
                        help='实验结束后生成全图分类图（默认开启）')
    parser.add_argument('--no_draw_map', dest='draw_map', action='store_false',
                        help='关闭全图分类图生成')
    args = parser.parse_args()
    requested_dataset = args.dataset
    args.dataset = canonical_dataset_name(args.dataset)
    if args.samples_per_class is None:
        args.samples_per_class = get_default_samples_per_class(args.dataset)
    args.save_dir = resolve_save_dir(args.save_dir)
    if args.num_runs < 1 or args.num_runs > len(DSFORMER_SEEDS):
        raise ValueError(
            f'DSFormer 仅提供 {len(DSFORMER_SEEDS)} 个固定种子，'
            f'当前 num_runs={args.num_runs}'
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[*] Acceleration: AMP={args.amp}, TF32={args.tf32}, "
          f"DataLoader workers={args.num_workers}")
    if requested_dataset != args.dataset:
        print(f"[*] Dataset alias: {requested_dataset} -> {args.dataset}")
    print(f"🚀 Device: {device} (Physical GPU ID: {args.device})")
    print(f"   Dataset : {args.dataset}  |  Model: {args.model}")
    print(f"   Epochs  : {args.epochs}   |  SPC  : {args.samples_per_class}")
    print(f"   Save dir: {args.save_dir}")
    print("\n[*] Loading HSI data and computing PCA...")
    raw_image, gt = load_hsi_data(args.dataset, args.dataset_dir)
    num_classes = int(gt.max()) + 1
    num_bands   = raw_image.shape[2]
    pca_image   = apply_pca(raw_image, n_components=1)
    print(f"    Image shape : {raw_image.shape}")
    print(f"    Num classes : {num_classes}")
    print(f"    Num bands   : {num_bands}")
    all_oa, all_aa, all_kappa, all_class_acc = [], [], [], []
    all_confusion_matrices = []
    all_train_time, all_test_time = [], []
    profile_info = None
    for run in range(1, args.num_runs + 1):
        seed = DSFORMER_SEEDS[run - 1]
        set_random_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = args.tf32
            torch.backends.cudnn.allow_tf32 = args.tf32
            torch.set_float32_matmul_precision('high' if args.tf32 else 'highest')
        print_run_header(run, args.num_runs, seed,
                         args.dataset, args.samples_per_class)
        train_gt, test_gt = sample_train_test_dsformer(
            gt, args.samples_per_class, seed=seed
        )
        split_path = None
        if args.save_splits:
            split_dir = os.path.join(args.save_dir, 'splits', 'dsformer', args.dataset)
            os.makedirs(split_dir, exist_ok=True)
            split_path = os.path.join(split_dir, f'run{run:02d}.npz')
            np.savez_compressed(
                split_path,
                train_gt=train_gt,
                test_gt=test_gt,
                seed=np.asarray(seed, dtype=np.int64),
                train_samples_per_class=np.asarray(args.samples_per_class, dtype=np.int64),
            )
            print(f"    DSFormer split saved: {split_path}")
        else:
            print("    DSFormer split: generated in memory (not saved; use --save_splits to keep .npz)")
        train_dataset = HSIPatchDataset(
            raw_image, train_gt,
            patch_size=args.patch_size,
            pca_image=pca_image,
            data_aug=True
        )
        test_dataset = HSIPatchDataset(
            raw_image, test_gt,
            patch_size=args.patch_size,
            pca_image=pca_image,
            data_aug=False
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
        class_weights = compute_class_weights(train_gt, num_classes)
        model = create_model(args, num_classes, num_bands).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Params: {total_params:,}")
        if profile_info is None:
            profile_inputs = (
                torch.zeros(1, num_bands, args.patch_size, args.patch_size, device=device),
                torch.zeros(1, 1, args.patch_size, args.patch_size, device=device),
            )
            profile_info = profile_model(model, profile_inputs)
            print(f"    FLOPs : {profile_info.flops} ({profile_info.status})")
        criterion = FocalOrthogonalLoss(
            num_classes=num_classes,
            gamma=2.0,
            lambda_ortho=args.lambda_ortho,
            ignore_index=-1,
            class_weights=class_weights,
        ).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=args.amp and device.type == 'cuda'
        )
        def lr_lambda(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        run_train_time = 0.0
        save_path = os.path.join(
            args.save_dir, f"{args.model}_{args.dataset}_run{run}.pth"
        )
        history_path = os.path.join(
            args.save_dir, f"history_{args.model}_{args.dataset}_run{run}.jsonl"
        )
        history_f = open(history_path, 'w', encoding='utf-8')
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss_info = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, args,
                scaler=scaler
            )
            epoch_time = time.time() - t0
            run_train_time += epoch_time
            scheduler.step()
            avg_loss = loss_info['total']
            current_lr = scheduler.get_last_lr()[0]
            epoch_record = {
                'epoch':      epoch,
                'lr':         current_lr,
                'epoch_time': round(epoch_time, 3),
                'loss_total': round(loss_info['total'], 6),
                'loss_main':  round(loss_info['main'],  6),
                'loss_ortho': round(loss_info['ortho'], 6),
                'weighted_main': round(loss_info['main'], 6),
                'weighted_ortho': round(criterion.lambda_ortho * loss_info['ortho'], 6),
                'lambda_ortho_effective': criterion.lambda_ortho,
                'grad_norm': round(loss_info['grad_norm'], 6),
            }
            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                print(f"    Epoch {epoch:>3d}/{args.epochs}"
                      f"  loss={avg_loss:.4f}"
                      f"  (m={loss_info['main']:.3f}"
                      f"/o={loss_info['ortho']:.3f})"
                      f"  lr={current_lr:.2e}")
            history_f.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
            history_f.flush()
        history_f.close()
        torch.save(model.state_dict(), save_path)
        print(f"\n    Final checkpoint saved @ epoch {args.epochs}")
        print(f"    History → {history_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
        t1 = time.time()
        test_results = evaluate(model, test_loader, device, args)
        test_time = time.time() - t1
        test_oa    = test_results["Accuracy"]
        test_aa    = test_results["AA"]
        test_kappa = test_results["Kappa"]
        test_cacc  = test_results["class acc"]
        test_cm    = np.asarray(test_results["Confusion matrix"], dtype=np.int64)
        print(f"\n    ── Test Results ──")
        print(f"    OA    = {test_oa:.2f}%")
        print(f"    AA    = {test_aa:.2f}%")
        print(f"    Kappa = {test_kappa:.2f}%")
        print("    Confusion matrix (rows=true, columns=predicted):")
        print(test_cm)
        all_oa.append(test_oa)
        all_aa.append(test_aa)
        all_kappa.append(test_kappa)
        all_class_acc.append(test_cacc)
        all_confusion_matrices.append(test_cm)
        all_train_time.append(run_train_time)
        all_test_time.append(test_time)
        with open(history_path, 'a', encoding='utf-8') as f:
            summary_record = {
                'epoch':         'final',
                'final_epoch':   args.epochs,
                'split_path':    split_path,
                'test_oa':       round(test_oa, 4),
                'test_aa':       round(test_aa, 4),
                'test_kappa':    round(test_kappa, 4),
                'confusion_matrix': test_cm.tolist(),
                'train_time_s':  round(run_train_time, 2),
                'test_time_s':   round(test_time, 2),
            }
            f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
        if args.draw_map:
            map_path = os.path.join(
                args.save_dir,
                f"map_{args.model}_{args.dataset}_run{run}.png"
            )
            inference_and_draw(
                model, raw_image, pca_image, gt,
                args.patch_size, num_classes, device, args, map_path
            )
    oa_mean, oa_std       = np.mean(all_oa),    np.std(all_oa)
    aa_mean, aa_std       = np.mean(all_aa),    np.std(all_aa)
    kappa_mean, kappa_std = np.mean(all_kappa), np.std(all_kappa)
    stacked_class_acc = np.stack(all_class_acc, axis=0)
    mean_class_acc = np.mean(stacked_class_acc, axis=0)
    std_class_acc = np.std(stacked_class_acc, axis=0)
    aggregate_cm = np.sum(np.stack(all_confusion_matrices, axis=0), axis=0)
    cm_header = "          " + "".join(f"P{i + 1:>7d}" for i in range(num_classes))
    cm_rows = [
        f"  T{i + 1:>3d}  " + "".join(f"{int(v):8d}" for v in row)
        for i, row in enumerate(aggregate_cm)
    ]
    class_names = DATASET_CLASS_NAMES.get(
        args.dataset,
        [f"Class {i + 1}" for i in range(num_classes)]
    )
    summary_lines = [
        "=" * 60,
        "  EXPERIMENT SUMMARY",
        "=" * 60,
        f"  Dataset : {args.dataset}",
        f"  Model   : {args.model}",
        f"  Runs    : {args.num_runs}",
        f"  SPC     : {args.samples_per_class}",
        "  Split   : DSFormer fixed-count (no validation; remaining labels are test)",
        f"  Checkpoint: final epoch {args.epochs}",
        f"  Params  : {profile_info.params:,}",
        f"  FLOPs   : {profile_info.flops} ({profile_info.status})",
        "",
        f"  OA    : {oa_mean:.2f} ± {oa_std:.2f} %",
        f"  AA    : {aa_mean:.2f} ± {aa_std:.2f} %",
        f"  Kappa : {kappa_mean:.2f} ± {kappa_std:.2f} %",
        "",
        "  Per-run OA:",
    ] + [f"    Run {i+1:>2d}: {v:.2f}%" for i, v in enumerate(all_oa)] + [
        "",
        "  Mean per-class accuracy:",
        format_class_acc_with_std(mean_class_acc, std_class_acc, class_names),
        "",
        "  Aggregate confusion matrix (rows=true, columns=predicted):",
        cm_header,
        *cm_rows,
        "",
        f"  Mean train time : {np.mean(all_train_time):.1f} s",
        f"  Mean test  time : {np.mean(all_test_time):.1f} s",
        "=" * 60,
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    log_file = os.path.join(
        args.save_dir, f"results_{args.model}_{args.dataset}.txt"
    )
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("--- Experiment Config ---\n")
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False))
        f.write("\n\n")
        f.write(summary)
        f.write("\n")
    json_file = os.path.join(
        args.save_dir, f"results_{args.model}_{args.dataset}.json"
    )
    json_data = {
        "config": vars(args),
        "split_protocol": "dsformer_fixed_count",
        "seeds": list(DSFORMER_SEEDS[:args.num_runs]),
        "checkpoint_selection": "final_epoch",
        "params": profile_info.params,
        "flops": profile_info.flops,
        "flops_status": profile_info.status,
        "OA_mean": round(oa_mean, 4),
        "OA_std":  round(oa_std,  4),
        "AA_mean": round(aa_mean, 4),
        "AA_std":  round(aa_std,  4),
        "Kappa_mean": round(kappa_mean, 4),
        "Kappa_std":  round(kappa_std,  4),
        "per_run_OA": [round(v, 4) for v in all_oa],
        "per_run_confusion_matrices": [cm.tolist() for cm in all_confusion_matrices],
        "aggregate_confusion_matrix": aggregate_cm.tolist(),
        "mean_class_acc": [round(v, 4) for v in mean_class_acc.tolist()],
        "mean_train_time_s": round(float(np.mean(all_train_time)), 2),
        "mean_test_time_s":  round(float(np.mean(all_test_time)),  2),
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f"\n🎉 All {args.num_runs} runs finished.")
    print(f"   Log  → {log_file}")
    print(f"   JSON → {json_file}")


if __name__ == '__main__':
    main()
