import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# =====================================================
# IMAGE SELECTION
# =====================================================

print("Select a leaf image")

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Leaf Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_path:
    print("No image selected")
    exit()

image = cv2.imread(image_path)

if image is None:
    print("Could not load image")
    exit()

# Resize for consistency
height, width = image.shape[:2]

max_dim = 900

if max(height, width) > max_dim:
    scale = max_dim / max(height, width)

    image = cv2.resize(
        image,
        (
            int(width * scale),
            int(height * scale)
        )
    )

# =====================================================
# ROI SELECTION
# =====================================================

print("\nDraw a box around the leaf.")
print("Press ENTER when done.")

roi = cv2.selectROI(
    "Select Leaf",
    image,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyWindow("Select Leaf")

x, y, w, h = roi

if w == 0 or h == 0:
    print("No leaf selected")
    exit()

# =====================================================
# GRABCUT
# =====================================================

print("Running GrabCut...")

mask = np.zeros(image.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv2.grabCut(
    image,
    mask,
    (x, y, w, h),
    bgdModel,
    fgdModel,
    8,
    cv2.GC_INIT_WITH_RECT
)

leaf_mask = np.where(
    (mask == cv2.GC_FGD) |
    (mask == cv2.GC_PR_FGD),
    255,
    0
).astype(np.uint8)

# =====================================================
# MASK CLEANUP
# =====================================================

kernel = np.ones((5, 5), np.uint8)

leaf_mask = cv2.morphologyEx(
    leaf_mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

leaf_mask = cv2.morphologyEx(
    leaf_mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

# =====================================================
# FIND TARGET CONTOUR
# =====================================================

contours, _ = cv2.findContours(
    leaf_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"Contours found: {len(contours)}")

# If GrabCut fails, fall back to ROI rectangle
if len(contours) == 0:

    print("GrabCut contour extraction failed.")
    print("Falling back to ROI mask.")

    leaf_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.rectangle(
        leaf_mask,
        (x, y),
        (x + w, y + h),
        255,
        -1
    )

    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
if not contours:
    print("No contour found")
    exit()

roi_center_x = x + w // 2
roi_center_y = y + h // 2

best_contour = None
best_distance = float("inf")

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 500:
        continue

    M = cv2.moments(contour)

    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    distance = (
        (cx - roi_center_x) ** 2 +
        (cy - roi_center_y) ** 2
    )

    if distance < best_distance:
        best_distance = distance
        best_contour = contour

if best_contour is None:
    print("Leaf could not be isolated")
    exit()

clean_mask = np.zeros_like(leaf_mask)

cv2.drawContours(
    clean_mask,
    [best_contour],
    -1,
    255,
    cv2.FILLED
)

leaf_mask = clean_mask

# =====================================================
# EXTRACT LEAF
# =====================================================

isolated_leaf = cv2.bitwise_and(
    image,
    image,
    mask=leaf_mask
)

leaf_pixels = cv2.countNonZero(leaf_mask)

# =====================================================
# HSV ANALYSIS - HEALTHY GREEN DETECTOR
# =====================================================

print("Analyzing soot coverage...")

hsv = cv2.cvtColor(isolated_leaf, cv2.COLOR_BGR2HSV)

# Healthy leaf colors
lower_green = np.array([25, 30, 30])
upper_green = np.array([95, 255, 255])

healthy_mask = cv2.inRange(
    hsv,
    lower_green,
    upper_green
)

# Restrict to leaf area only
healthy_mask = cv2.bitwise_and(
    healthy_mask,
    healthy_mask,
    mask=leaf_mask
)

kernel_small = np.ones((3, 3), np.uint8)

healthy_mask = cv2.morphologyEx(
    healthy_mask,
    cv2.MORPH_OPEN,
    kernel_small
)

healthy_mask = cv2.morphologyEx(
    healthy_mask,
    cv2.MORPH_CLOSE,
    kernel_small
)

# Everything NOT healthy green = soot
soot_mask = cv2.bitwise_and(
    leaf_mask,
    cv2.bitwise_not(healthy_mask)
)

soot_mask = cv2.morphologyEx(
    soot_mask,
    cv2.MORPH_OPEN,
    kernel_small
)

soot_mask = cv2.morphologyEx(
    soot_mask,
    cv2.MORPH_CLOSE,
    kernel_small
)

# =====================================================
# CALCULATE SOOT %
# =====================================================

soot_pixels = cv2.countNonZero(soot_mask)

soot_percentage = (
    soot_pixels / leaf_pixels * 100
    if leaf_pixels > 0
    else 0
)

# =====================================================
# TEXTURE ANALYSIS
# =====================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

laplacian_var = cv2.Laplacian(
    gray,
    cv2.CV_64F
).var()

# =====================================================
# ENVIRONMENT MODEL
# =====================================================

STOMATAL_FACTOR = 1.75
BASE_CO2 = 22.0

efficiency_drop = min(
    soot_percentage * STOMATAL_FACTOR,
    100
)

estimated_efficiency = 100 - efficiency_drop

lost_carbon = (
    efficiency_drop / 100
) * BASE_CO2

# =====================================================
# STATUS
# =====================================================

if estimated_efficiency > 80:
    status = "HEALTHY"
elif estimated_efficiency > 50:
    status = "WARNING"
else:
    status = "CRITICAL"

# =====================================================
# OVERLAY
# =====================================================

overlay = image.copy()

overlay[soot_mask > 0] = [0, 0, 255]

# =====================================================
# SAVE RESULTS
# =====================================================

os.makedirs("results", exist_ok=True)

cv2.imwrite(
    "results/leaf_mask.png",
    leaf_mask
)

cv2.imwrite(
    "results/soot_mask.png",
    soot_mask
)

cv2.imwrite(
    "results/overlay.png",
    overlay
)

# =====================================================
# REPORT
# =====================================================

print("\n========== LEAF SOOT REPORT ==========")
print(f"Leaf Area              : {leaf_pixels}")
print(f"Soot Coverage          : {soot_percentage:.2f}%")
print(f"Texture Variance       : {laplacian_var:.2f}")
print(f"Tree Efficiency        : {estimated_efficiency:.2f}%")
print(f"Lost Carbon Capacity   : {lost_carbon:.2f} kg/year")
print(f"Status                 : {status}")
print("======================================")

# =====================================================
# DISPLAY
# =====================================================

cv2.imshow("Original", image)
cv2.imshow("Leaf Mask", leaf_mask)
cv2.imshow("Soot Mask", soot_mask)
cv2.imshow("Overlay", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()