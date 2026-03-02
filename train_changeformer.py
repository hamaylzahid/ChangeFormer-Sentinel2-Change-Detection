# put main training loop here
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from utils import initialize_ee, get_sentinel_image, ee_to_numpy, compute_indices, extract_patches
from datasets import ChangeDataset
from models import ChangeFormer
from losses_metrics import total_loss, compute_metrics
import ee

if __name__=="__main__":
    os.makedirs("outputs", exist_ok=True)
    initialize_ee()

    roi = ee.Geometry.Rectangle([72.95,19.05,73.05,19.15])
    img1 = get_sentinel_image("2017-01-01","2017-12-31",roi)
    img2 = get_sentinel_image("2023-01-01","2023-12-31",roi)

    arr1 = compute_indices(ee_to_numpy(img1, roi))
    arr2 = compute_indices(ee_to_numpy(img2, roi))

    patches1 = extract_patches(arr1, patch_size=256, stride=128)
    patches2 = extract_patches(arr2, patch_size=256, stride=128)

    print("Number of patches:", len(patches1), len(patches2))

    dataset = ChangeDataset(patches1, patches2, ndvi_thresh=0.01, augment=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChangeFormer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 20
    losses=[]
    for epoch in range(epochs):
        model.train()
        total=0
        for t1,t2,mask in tqdm(loader):
            t1,t2,mask = t1.to(device), t2.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(t1,t2)
            loss = total_loss(pred,mask)
            loss.backward()
            optimizer.step()
            total+=loss.item()
        losses.append(total/len(loader))
        print(f"Epoch {epoch+1}/{epochs} Loss: {losses[-1]:.4f}")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("outputs/loss_curve.png")
    plt.close()

    # Evaluation
    model.eval()
    with torch.no_grad():
        t1,t2,mask = next(iter(loader))
        t1,t2,mask = t1.to(device), t2.to(device), mask.to(device)
        pred = model(t1,t2)
        iou, precision, recall, f1 = compute_metrics(pred,mask)
        print(f"IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        plt.imshow(pred[0][0].cpu(), cmap="gray")
        plt.savefig("outputs/prediction_map.png")
        plt.close()