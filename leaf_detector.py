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
# 2. THE UPGRADED COLOR MASKING LOOP (HSV)
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
soot_mask = cv2.bitwise_not(healthy_green_mask)

# (Optional Texture Variance for the report)
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

# 5. Count how many pixels are flagged as pollution
soot_pixels = cv2.countNonZero(soot_mask)
total_leaf_pixels = gray.shape[0] * gray.shape[1]

# 6. Math time: Find out what percentage of the image is covered in pollution
soot_saturation_ratio = soot_pixels / total_leaf_pixels
soot_saturation_percentage = soot_saturation_ratio * 100

print(f"📊 Texture Variance Score: {laplacian_var:.2f}")
print(f"📊 Pollution Saturation Detected: {soot_saturation_percentage:.2f}%")




# 3. THE BOTANICAL ESTIMATION LOGIC
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





# 4.REPORT GENERATION
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
print(f" Calculated Soot Coverage  : {soot_saturation_percentage:.2f}% of total surface area")
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

# Show the original image and the black-and-white Soot Mask side-by-side
cv2.imshow("Original Leaf Image", resized_image)
cv2.imshow("Soot Detection Map (White = Soot)", soot_mask)

# Keep the windows open until you press any key on your keyboard
cv2.waitKey(0)
cv2.destroyAllWindows()