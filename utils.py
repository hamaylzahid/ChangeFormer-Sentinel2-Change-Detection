import ee
import geemap
import numpy as np
import random

# -----------------------------
# Google Earth Engine Setup
# -----------------------------
def initialize_ee():
    try:
        ee.Initialize(project='earth-engine-colab-489009')
        print("Earth Engine initialized!")
    except Exception as e:
        print("Authenticating Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project='earth-engine-colab-489009')
        print("Earth Engine initialized after authentication.")

# -----------------------------
# Data Acquisition
# -----------------------------
def get_sentinel_image(start_date, end_date, roi):
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B2", "B3", "B4", "B8", "B11"])
    )
    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(f"No images found for {start_date}-{end_date}.")
    print(f"Number of images: {count}")
    image = collection.median().clip(roi)
    return image

def ee_to_numpy(image, roi, scale=30):
    try:
        url = image.getDownloadURL({"scale": scale, "region": roi, "format": "NPY"})
        file_path = geemap.download_file(url, overwrite=True)
        arr = np.load(file_path)

        if arr.dtype.names:
            bands = arr.dtype.names
            h, w = arr.shape
            arr_np = np.zeros((h, w, len(bands)), dtype=np.float32)
            for i, b in enumerate(bands):
                arr_np[:, :, i] = arr[b]
            arr = arr_np

        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]

        print(f"Array shape: {arr.shape}, min: {arr.min()}, max: {arr.max()}")
        return arr.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"EE to numpy failed: {e}")

# -----------------------------
# Preprocessing
# -----------------------------
def compute_indices(img):
    B2, B3, B4, B8, B11 = img[:, :, 0:5].transpose(2,0,1)
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    ndbi = (B11 - B8) / (B11 + B8 + 1e-6)
    stacked = np.stack([B2, B3, B4, B8, ndvi, ndbi], axis=2)
    return stacked.astype(np.float32)

def extract_patches(img, patch_size=128, stride=64):
    patches = []
    h, w, _ = img.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h or pad_w:
        img = np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode='constant')
    h, w, _ = img.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patches.append(img[i:i+patch_size, j:j+patch_size])
    return patches

def augment_patch(t1, t2, mask):
    if random.random() > 0.5:
        t1, t2, mask = np.flip(t1, axis=0), np.flip(t2, axis=0), np.flip(mask, axis=0)
    if random.random() > 0.5:
        t1, t2, mask = np.flip(t1, axis=1), np.flip(t2, axis=1), np.flip(mask, axis=1)
    k = random.randint(0,3)
    t1, t2, mask = np.rot90(t1,k,(0,1)), np.rot90(t2,k,(0,1)), np.rot90(mask,k,(0,1))
    factor = 0.9 + 0.2 * random.random()
    t1[:,:,:4] = np.clip(t1[:,:,:4]*factor,0,1)
    t2[:,:,:4] = np.clip(t2[:,:,:4]*factor,0,1)
    shift_x, shift_y = random.randint(-4,4), random.randint(-4,4)
    t1 = np.roll(t1, shift_x, 0); t1 = np.roll(t1, shift_y, 1)
    t2 = np.roll(t2, shift_x, 0); t2 = np.roll(t2, shift_y, 1)
    mask = np.roll(mask, shift_x,0); mask = np.roll(mask, shift_y,1)
    return t1.copy(), t2.copy(), mask.copy()