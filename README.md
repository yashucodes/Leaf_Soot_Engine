# 🍃 Leaf Soot Engine

An advanced computer vision pipeline built with Python and OpenCV that automatically isolates a leaf from complex backgrounds and quantifies surface soot, dust, and particulate pollution using localized color-spectrum tracking.

---

## 🚀 Project Overview

Environmental monitoring often relies on manual observation or expensive chemical tests to assess air quality. This project introduces a lightweight, automated digital inspection tool. 

By utilizing **GrabCut foreground extraction** alongside **HSV color-space inversion matrices**, this system isolates organic structures (leaves) from chaotic background noise (branches, sky, artifacts) and maps localized surface contamination with high precision.

---

## 🧠 Key Technical Features

* **Foreground Segregation (GrabCut):** Implements iterative graph-cut segmentation to cleanly detach the leaf from background elements sharing similar color signatures.
* **HSV Color Optimization:** Leverages the Hue-Saturation-Value color space to maintain high-fidelity tracking completely independent of localized shadows or variable lighting conditions.
* **Hybrid Mask Filtering:** Employs a custom bitwise logical mask pipeline to evaluate regions that are definitively "not healthy green" purely within the boundary geometry of the isolated leaf.
* **Quantitative Saturation Reporting:** Calculates pixel-density distribution ratios to return a precise, verifiable pollution saturation percentage.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Core Libraries:** * `OpenCV` (Advanced Image Processing & Graph Computations)
  * `NumPy` (High-Performance Multi-Dimensional Vector Mathematics)

---

## 📦 Installation & Setup

1. **Clone the repository to your local machine:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Leaf_soot_Engine.git](https://github.com/YOUR_USERNAME/Leaf_Soot_Engine.git)
   cd Lafe_Soot_Engine
