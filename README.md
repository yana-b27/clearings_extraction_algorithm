# clearings_extraction_algorithm

The algorithm is designed to automatically identify forest clearings under power lines in Sentinel-2 images. The algorithm consists of Logistic Regression model for selecting objects similar to forest clearings in terms of brightness (e.g. meadows and vegetation-free lands) into a separate class of low-growth vegetation and Probabilistic Hough Transform for finding linear structures on the binary mask of low-growth vegetation from the classification map.

A sample of about 15000 pixels was compiled in QGIS software and further rasterized. Before training the model, the images were pre-processed by applying value normalization and saturation contrasting to reduce the brightness of abnormally bright pixels.  Rasterized polygons were used to extract pixels for sampling. In addition to brightness features, features such as SAVI, NDWI (normalized difference water index) and NDBI (normalized difference built-up index) spectral indices were applied in the spectral channels of summer and winter Sentinel-2 images.

An experiment was conducted and the classification quality metrics accuracy, f1 score were calculated to identify the machine learning model that best selects the low vegetation class. Logistic regression was chosen as the fastest and highest quality model for our task.

The Probabilistic Hough Transform was chosen as the line search algorithm. Before its application, a Gaussian filter is applied to the binary mask of the low-growth vegetation class to distort objects that do not belong to forest clearings, while linear clearings retain linearity. Next, object boundaries were found using binary erosion. Based on the obtained mask of object boundaries, Hough Transform finds lines. Selected lines belong to forest clearings.

The project contains:
- code of the algorithm in `clearings_extraction_algorithm.py`
- example of how to use the code in `example.ipynb`
- satellite images in `.tiff` format stored in `images` folder
- set of polygons for each experimental area in shapefile format stored in `polygons_sets` folder
- data of power lines stored in `.shp` as line geometry in `power_lines` folder
- data of power towers stored in `.shp` as point geometry in `power_towers` folder
- results of clearings extraction algorithm - masks of power line clearings in `.tiff` format stored in `results` folder
- some plots of visualization of results in `svg_plots` folder
