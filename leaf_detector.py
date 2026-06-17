import cv2
import numpy as np
import tkinter as tk                  # Built-in tool to make pop-up windows
from tkinter import filedialog        # Specifically for choosing files

# 1. IMAGE PRE-PROCESSING
print("Please choose a leaf image.")
root = tk.Tk() # Hiding the main blank window that tkinter makes
root.withdraw()

image_path = filedialog.askopenfilename( # opening file explorer for user to select an image
    title="Select Urban Leaf Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)
if not image_path:
    print("Error: No file was selected. Exiting engine.")
    exit()

print(f"Selected file: {image_path}")
print("Loading urban leaf sample...")
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (800, 800)) # Resize the image to 800x800 pixels so the math remains uniform
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
print("Image successfully loaded, resized, and converted to grayscale!")

# ==========================================
# 2. GRABCUT FOREGROUND EXTRACTION (NEW)
# ==========================================
# Goal: isolate ONLY the leaf from the background, so that anything
# outside the leaf's silhouette (other leaves, soil, blur, dark gaps)
# never gets a chance to be counted as soot.

print("✂️  Running GrabCut to isolate the leaf from the background...")

# GrabCut needs an initial rectangle that roughly contains the foreground object.
# We use a margin inset from the edges, assuming the leaf is roughly centered
# and fills most of the frame (true for most close-up leaf photos).
h, w = resized_image.shape[:2]
margin_x, margin_y = int(w * 0.04), int(h * 0.04)
grabcut_rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

# Mask that GrabCut will fill in with foreground/background labels
gc_mask = np.zeros((h, w), dtype=np.uint8)

# Internal models GrabCut uses while iterating (required by the function, not used after)
bgd_model = np.zeros((1, 65), dtype=np.float64)
fgd_model = np.zeros((1, 65), dtype=np.float64)

cv2.grabCut(
    resized_image,
    gc_mask,
    grabcut_rect,
    bgd_model,
    fgd_model,
    5,                      # number of iterations; 5 is a good speed/accuracy tradeoff
    cv2.GC_INIT_WITH_RECT
)

# GrabCut labels pixels as: 0=bg, 1=fg, 2=probable bg, 3=probable fg
# We treat both "fg" and "probable fg" as part of the leaf.
leaf_foreground_mask = np.where(
    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
    255,
    0
).astype(np.uint8)

# Clean up small holes/noise in the foreground mask with morphological closing
kernel = np.ones((7, 7), np.uint8)
leaf_foreground_mask = cv2.morphologyEx(leaf_foreground_mask, cv2.MORPH_CLOSE, kernel)
leaf_foreground_mask = cv2.morphologyEx(leaf_foreground_mask, cv2.MORPH_OPEN, kernel)

leaf_pixel_count = cv2.countNonZero(leaf_foreground_mask)
print(f"✂️  Leaf silhouette isolated: {leaf_pixel_count} pixels identified as leaf area.")

# ==========================================
# 3. THE UPGRADED COLOR MASKING LOOP (HSV)
# ==========================================

print("🔬 Running HSV color spectrum analysis...")

# 1. Convert the image to HSV (Hue, Saturation, Value) to ignore shadows
hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

# 2. Define the color range for "Healthy Green"
# These numbers tell the computer exactly what shades of green to look for
lower_green = np.array([35, 40, 40])
upper_green = np.array([90, 255, 255])

# 3. Create a mask that ONLY highlights the healthy green parts of the leaf
healthy_green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# 4. The Magic Trick: Flip the mask! (Invert it)
# If a pixel is NOT healthy green, we flag it as dust/pollution
soot_mask_raw = cv2.bitwise_not(healthy_green_mask)

# 5. NEW: Constrain the soot mask to ONLY the leaf area found by GrabCut.
# This is the actual fix -- bitwise_and forces every pixel outside the
# leaf silhouette to 0 (black), no matter what color it was.
soot_mask = cv2.bitwise_and(soot_mask_raw, soot_mask_raw, mask=leaf_foreground_mask)

# (Optional Texture Variance for the report)
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

# 6. Count how many pixels are flagged as pollution
soot_pixels = cv2.countNonZero(soot_mask)

# 7. IMPORTANT: total area is now the LEAF area, not the whole 800x800 frame.
# Otherwise the percentage is artificially diluted by background pixels
# that were never part of the leaf to begin with.
total_leaf_pixels = leaf_pixel_count if leaf_pixel_count > 0 else (gray.shape[0] * gray.shape[1])

# 8. Math time: Find out what percentage of the LEAF is covered in pollution
soot_saturation_ratio = soot_pixels / total_leaf_pixels
soot_saturation_percentage = soot_saturation_ratio * 100

print(f"📊 Texture Variance Score: {laplacian_var:.2f}")
print(f"📊 Pollution Saturation Detected: {soot_saturation_percentage:.2f}%")




# 4. THE BOTANICAL ESTIMATION LOGIC
print("Calculating environmental impact...")

# Environmental constants (our baseline science numbers)
STOMATAL_DENSITY_FACTOR = 1.75 # How aggressively soot blocks this tree's breathing pores
HEALTHY_BASELINE_EFFICIENCY = 100.0 # A perfectly clean tree operates at 100%
BASE_CARBON_ABSORPTION = 22.0 # A healthy mature tree absorbs about 22 kg of CO2 per year

efficiency_drop = soot_saturation_percentage * STOMATAL_DENSITY_FACTOR # % drop in carbon inhalation efficiency
if efficiency_drop > 100.0: # a tree can't lose more than 100% of its breathing ability, so we cap it at 100%
    efficiency_drop = 100.0
estimated_efficiency = HEALTHY_BASELINE_EFFICIENCY - efficiency_drop # remaining carbon inhalation efficiency
net_lost_carbon = (efficiency_drop / 100.0) * BASE_CARBON_ABSORPTION # net kilograms of CO2 lost per tree per year

print("Botanical impact formulas successfully calculated!")





# 5. REPORT GENERATION
# Determine the status based on the remaining efficiency
if estimated_efficiency > 80:
    status = "HEALTHY (Optimal Inhalation)"
elif estimated_efficiency > 50:
    status = "WARNING (Moderate Suffocation)"
else:
    status = "CRITICAL (Severe Suffocation)"

#ASCII Dashboard Report
print("================= CANOPY SOOT SATURATION REPORT =================")
print(" Analysis Engine: Laplacian Surface Texture Variance V1.0")
print("-----------------------------------------------------------------")
print("IMAGE PROCESSING METRICS:")
print(f" Surface Texture Variance : {laplacian_var:.2f}")
print(f" Leaf Area Isolated (GrabCut): {leaf_pixel_count} px")
print(f" Calculated Soot Coverage  : {soot_saturation_percentage:.2f}% of LEAF surface area")
print("")
print("CLIMATE DEGRADATION IMPACT:")
print(f" Stomatal Suffocation Index: {status}")
print(f" Estimated CO2 Inhalation  : {estimated_efficiency:.1f}% of baseline potential")
print(f" Net Lost Carbon Capacity  : -{net_lost_carbon:.2f} kg CO2 / tree / year")
print("")
print("MANAGEMENT ACTION ADVISORY:")
if estimated_efficiency < 70:
    print(" Urban canopy in this sector is choked. Recommend immediate")
    print(" automated water-mist spraying to wash leaves and restore capacity.")
else:
    print("Canopy is functioning well. Continue routine monitoring.")
print("=================================================================\n")


#VISUALIZING THE DETECTIVE WORK
print("Opening visual display windows... Press any key to close them.")

# Show the original image, the GrabCut leaf silhouette, and the final
# background-suppressed Soot Mask side-by-side
cv2.imshow("Original Leaf Image", resized_image)
cv2.imshow("GrabCut Leaf Silhouette (White = Leaf)", leaf_foreground_mask)
cv2.imshow("Soot Detection Map (White = Soot, Background Suppressed)", soot_mask)

# Keep the windows open until you press any key on your keyboard
cv2.waitKey(0)
cv2.destroyAllWindows()