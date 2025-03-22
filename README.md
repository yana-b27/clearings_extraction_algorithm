# clearings_extraction_algorithm

The project is devoted to the creation of an algorithm to identify forest clearings under power lines on Sentinel-2 satellite images. The algorithm uses logistic regression model to identify forest clearings based on brightness of image pixels in different spectral channels, and Hough probabilistic transform finds forest clearings based on linearity feature. Thus, the developed algorithm finds the forest clearings under power lines based on its understandable by humans features, so the algorithm and its results are interpretable.

The project contains:
- code of the algorithm in `clearings_extraction_algorithm.py`
- example of how to use the code in `example.ipynb`
- satellite images in `.tiff` format stored in `images` folder
- set of polygons for each experimental area in shapefile format stored in `polygons_sets` folder
- data of power lines stored in `.shp` as line geometry in `power_lines` folder
- data of power towers stored in `.shp` as point geometry in `power_towers` folder
- results of clearings extraction algorithm - masks of power line clearings in `.tiff` format stored in `results` folder
- some plots of visualization of results in `svg_plots` folder
