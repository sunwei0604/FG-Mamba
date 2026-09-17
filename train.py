import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import metrics
import matplotlib.pyplot as plt
import os
from scipy import io


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args,
                scaler=None):
    model.train()
    losses = []
    main_losses, ortho_losses = [], []
    grad_norms = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
    for patch_raw, patch_pca, targets in pbar:
        patch_raw = patch_raw.to(device, non_blocking=True)
        patch_pca = patch_pca.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and scaler.is_enabled()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(patch_raw, patch_pca)
            loss, loss_dict = criterion(logits, targets, model.embedding.proj.weight)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip if args.grad_clip > 0 else float('inf')
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip if args.grad_clip > 0 else float('inf')
            )
            optimizer.step()
        grad_norms.append(float(grad_norm))
        losses.append(loss.item())
        main_losses.append(loss_dict.get('main', 0.0))
        ortho_losses.append(loss_dict.get('ortho', 0.0))
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
    return {
        'total': float(np.mean(losses)),
        'main':  float(np.mean(main_losses)),
        'ortho': float(np.mean(ortho_losses)),
        'grad_norm': float(np.mean(grad_norms)),
    }


def evaluate(model, test_loader, device, args):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="[Test]", leave=False)
        for patch_raw, patch_pca, targets in pbar:
            patch_raw = patch_raw.to(device, non_blocking=True)
            patch_pca = patch_pca.to(device, non_blocking=True)
            amp_enabled = getattr(args, 'amp', False) and device.type == 'cuda'
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(patch_raw, patch_pca)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    results = metrics(all_preds, all_targets)
    return results


def inference_and_draw(model, raw_image, pca_image, gt, patch_size, num_classes, device, args, save_path):
    model.eval()
    H, W, C = raw_image.shape
    ps = patch_size // 2
    pad_raw = np.pad(raw_image, ((ps, ps), (ps, ps), (0, 0)), mode='reflect')
    if pca_image is not None:
        pad_pca = np.pad(pca_image, ((ps, ps), (ps, ps), (0, 0)), mode='reflect')
    else:
        pad_pca = None
    outputs = np.zeros((H, W), dtype=int)
    batch_raw = []
    batch_pca = []
    batch_coords = []
    print("[*] Running sliding window inference on FULL image (this may take longer)...")
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                patch_r = pad_raw[i: i + patch_size, j: j + patch_size].transpose(2, 0, 1)
                batch_raw.append(patch_r)
                if pad_pca is not None:
                    patch_p = pad_pca[i: i + patch_size, j: j + patch_size].transpose(2, 0, 1)
                    batch_pca.append(patch_p)
                batch_coords.append((i, j))
                if len(batch_raw) == args.batch_size or (i == H - 1 and j == W - 1):
                    if len(batch_raw) == 0:
                        continue
                    b_raw = torch.tensor(np.array(batch_raw), dtype=torch.float32).to(device)
                    b_pca = torch.tensor(np.array(batch_pca), dtype=torch.float32).to(device)
                    logits = model(b_raw, b_pca)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    for idx, (cx, cy) in enumerate(batch_coords):
                        outputs[cx, cy] = preds[idx] + 1
                    batch_raw.clear()
                    batch_pca.clear()
                    batch_coords.clear()
    palette = np.array([
        [0, 0, 0],
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        [0, 255, 255], [255, 0, 255], [176, 48, 96], [46, 139, 87],
        [160, 32, 240], [255, 127, 80], [127, 255, 212], [218, 112, 214],
        [160, 82, 45], [127, 255, 0], [216, 191, 216], [238, 0, 0],
        [139, 0, 0], [238, 154, 0], [85, 26, 139], [148, 204, 120],
        [188, 215, 78], [238, 234, 63], [244, 127, 33], [123, 18, 20]
    ])
    if num_classes > len(palette) - 1:
        cmap = plt.get_cmap('tab20', num_classes)
        palette_ext = (cmap(np.arange(num_classes))[:, :3] * 255).astype(int)
        palette = np.vstack([[[0, 0, 0]], palette_ext])
    palette = palette * 1.0 / 255.0
    rgb_canvas = np.zeros((H, W, 3))
    for i in range(1, num_classes + 1):
        mask = (outputs == i)
        rgb_canvas[mask, 0] = palette[i, 0]
        rgb_canvas[mask, 1] = palette[i, 1]
        rgb_canvas[mask, 2] = palette[i, 2]
    plt.figure(figsize=(W / 100, H / 100), dpi=900)
    plt.axis("off")
    plt.imshow(rgb_canvas)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    mat_path = os.path.splitext(save_path)[0] + ".mat"
    io.savemat(mat_path, {"prediction": outputs.astype(np.int16)})
    print(f"[*] Map generated and saved successfully to: {save_path}")
    print(f"[*] Prediction MAT saved successfully to: {mat_path}")
