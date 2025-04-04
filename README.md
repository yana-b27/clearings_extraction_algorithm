# clearings_extraction_algorithm

This algorithm automates the detection of forest clearings beneath power lines using Sentinel-2 satellite imagery. It integrates a Logistic Regression model to classify low-growth vegetation (e.g., meadows and vegetation-free areas) based on brightness and spectral features, followed by a Probabilistic Hough Transform to identify linear structures—indicative of clearings—within the classified binary mask.

## Methodology
A training dataset of approximately 15,000 pixels was created using QGIS, where polygons were rasterized to sample pixels. Pre-processing involved normalizing pixel values and applying saturation contrast to mitigate the impact of overly bright pixels. Beyond brightness, additional features were derived from spectral indices—SAVI (Soil-Adjusted Vegetation Index), NDWI (Normalized Difference Water Index), and NDBI (Normalized Difference Built-Up Index)—computed from summer and winter Sentinel-2 imagery.

To determine the optimal machine learning model, an experiment evaluated classification performance using accuracy and F1-score metrics. Logistic Regression emerged as the best performer, balancing speed and quality for isolating the low-growth vegetation class.

For line detection, the Probabilistic Hough Transform was employed. Prior to its application, a Gaussian filter smooths the binary mask of low-growth vegetation, reducing noise from non-clearing objects while preserving the linearity of clearings. Binary erosion then enhances object boundaries, enabling the Hough Transform to detect lines corresponding to forest clearings.

## Repository structure
```
.
├── __pycache__/                      # Compiled Python files
├── images/                           # Sentinel-2 images for algorithm testing
├── model/                            # Logistic Regression model file
├── polygons_sets/                    # Shapefiles of polygon sets created in QGIS
├── power_lines/                      # OpenStreetMap linear vector files of power lines
├── power_towers/                     # OpenStreetMap point vector files of power towers
├── results/                          # Georeferenced .tif files with algorithm outputs
├── svg_plots/                        # Visualizations from Jupyter notebooks
├── LICENSE                           # License file
├── README.md                         # Repository overview and instructions
├── calculate_metrics.ipynb           # Notebook demonstrating quality metrics creation and computation
├── choose_ml_algorithm.ipynb         # Notebook detailing ML algorithm selection
├── clearings_extraction_algorithm.py # Core algorithm implementation
├── requirements.txt                  # Python dependencies
└── use_hough_transform.ipynb         # Notebook showcasing Hough Transform for line detection
```
This repository provides a complete workflow—from data preparation and model training to line detection and result visualization—tailored for monitoring forest clearings under power lines.
