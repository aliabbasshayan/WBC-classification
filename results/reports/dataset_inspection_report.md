# Dataset Inspection and Verification Report: Raabin-WBC

## 1. Executive Summary
- **Dataset Title**: Raabin-WBC Double-Labeled Cropped Leukocyte Images
- **Total Images Identified**: 16,633 JPEG images
- **Target Cell Types (5 Classes)**:
  1. Basophil
  2. Eosinophil
  3. Lymphocyte
  4. Monocyte
  5. Neutrophil
- **Integrity Status**: 0 corrupted or zero-byte files; 0 exact file duplicates.

## 2. Split and Class Distribution Breakdown

| WBC Class | Train Count (%) | TestA Count (%) | TestB Count | Imbalance Ratio (vs Basophil) |
| :--- | :--- | :--- | :--- | :--- |
| **Basophil** | 212 (2.08%) | 89 (2.05%) | 0 | 1.0x (Minority) |
| **Eosinophil** | 744 (7.31%) | 322 (7.42%) | 0 | 3.5x |
| **Lymphocyte** | 2427 (23.85%) | 1034 (23.83%) | 148 | 11.4x |
| **Monocyte** | 561 (5.51%) | 234 (5.39%) | 0 | 2.6x |
| **Neutrophil** | 6231 (61.24%) | 2660 (61.3%) | 1971 | 29.4x (Majority) |
| **Total** | **10175** | **4339** | **2119** | - |

## 3. Key Observations and Methodological Considerations
1. **Extreme Class Imbalance**: Neutrophils comprise ~61.2% of the dataset, while Basophils represent only ~2.1%. Unweighted Cross-Entropy loss would bias predictions towards Neutrophils. The training pipeline must apply class-weighted loss (inverse class frequency or effective number of samples).
2. **Official Split Structure**:
   - `Train` (10,175 images) and `TestA` (4,339 images) exhibit identical class proportions.
   - `TestB` contains only Lymphocytes and Neutrophils (2,119 images) and is unsuitable for 5-class evaluation.
   - 26 slide IDs are shared between Train and TestA, demonstrating that the official Raabin-WBC split is an image-level stratified split rather than a patient-grouped split.
3. **Validation Strategy**:
   - `TestA` is preserved strictly as held-out test data.
   - A stratified validation split (15%, N=1,526) is created from `Train` (leaving 8,649 training images).
4. **Image Dimensions**:
   - Dimensions range from 575x575 to 575x575.
   - The median resolution is 575x575.
   - For CPU training efficiency and low RAM usage, images will be uniformly resized to 128x128 or 160x160 with bilinear interpolation and standard ImageNet normalization.

Generated automatically by Milestone 2 analysis pipeline.
