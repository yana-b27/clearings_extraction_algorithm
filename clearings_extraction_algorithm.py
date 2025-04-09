"""
clearings_extraction_algorithm.py

Module for extracting power line clearings from satellite imagery.

This module provides a set of functions and classes for detecting power line clearings
in satellite images. It includes methods for creating image datasets for exact areas,
learning a land classification model, and extracting clearings from land classification maps.

Classes:
    ImageDataset: A class representing a dataset of satellite images for exact area.
    LandClassificationModel: A class representing a land classification model.
    ClearingsExtractor: A class representing an extractor for power line clearings
        from satellite imagery.
    ClearingsDetectionMetrics: A class representing metrics for power line clearings detection.

Functions:
    rasterize_polygons: Rasterizes the given shapefile to the given raster, and writes
        the result to the given path.
    make_feature_matrix: Creates a feature matrix from the given dataset.

Notes:
    This module is designed to work with Sentinel-2 satellite images.
    The functions and classes in this module are intended to be used together to
    extract power line clearings from satellite imagery.
"""

import time
import joblib
from osgeo import ogr, gdal
import rioxarray as rxr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import earthpy.plot as ep
import spyndex
from PIL import Image, ImageDraw
from skimage.morphology import disk
from skimage.morphology import binary_opening, binary_erosion
from scipy.ndimage import gaussian_filter
from skimage.transform import probabilistic_hough_line
from scipy.ndimage import distance_transform_edt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# Color map for land classification map
classes = ["Водные объекты", "Леса", "Редколесья", "Луга", "Антропогенные объекты"]
colors = ["#2072AF", "#00563B", "#7BA05B", "#EEDC82", "#CC3333"]
class_map = ListedColormap(colors)


class ImageDataset:
    """
    A class representing a dataset of satellite images for exact area.

    Attributes:
      image_data_3d (3D numpy array): A 3D numpy array of size (height, width, number of channels) containing satellite image data.
      image_data_2d_arr (2D numpy array): A 2D numpy array of size (height*width, number of channels) containing satellite image data.

    Methods:
      add_channels(summer_image_path, winter_image_path): Adds channels from summer and winter images to the image dataset.
      saturation_contrast(): Applies contrast stretching to the satellite image channels.
      normalize_channels(num_channel_end, num_channel_start=0): Normalizes the channels of the satellite image data.
      compute_indices(): Computes spectral indices (SAVI, NDWI, NDBI) from the satellite image data.
      make_image_2d(): Creates a 2D numpy array from the 3D numpy array and normalizes the values.

    Notes:
      The `image_data_3d` attribute, which is a 3D numpy array, was created for channels visualization on the map.
      The `image_data_2d_arr` attribute is a 2D numpy array which was created for learning models.
    """

    def __init__(self, image_data_3d=None, image_data_2d_arr=None):
        """
        __init__ method for ImageDataset class.

        Parameters
        ----------
        image_data_3d : 3D numpy array
            3D numpy array of size (height, width, number of channels) containing satellite image data
        image_data_2d_arr : 2D numpy array
            2D numpy array of size (height*width, number of channels) containing satellite image data
        """
        self.image_data_3d = image_data_3d
        self.image_data_2d_arr = image_data_2d_arr

    def add_channels(self, summer_image_path, winter_image_path):
        """
        Method for adding channels from summer and winter images to image dataset.

        Parameters
        ----------
        summer_image_path : str
            path of the summer satellite image
        winter_image_path : str
            path of the winter satellite image
        """

        summer_image = rxr.open_rasterio(summer_image_path)
        winter_image = rxr.open_rasterio(winter_image_path)

        image_dataset = np.zeros(
            (summer_image.shape[1], summer_image.shape[2], 11), dtype=np.float64
        )

        for b in range(5):
            image_dataset[:, :, b] = summer_image[b, :, :]
        for b in range(5, 8):
            image_dataset[:, :, b] = winter_image[b - 5, :, :]

        self.image_data_3d = image_dataset

    def saturation_contrast(self):
        """
        Method for contrast stretching of satellite image channels.
        """

        def change_max_1(x):
            """
            A function for changing the maximum value of a numpy array to a 99% percentile.

            Parameters
            ----------
            x : numpy array
                The input array

            Returns
            -------
            numpy array
                The array with the maximum value changed to the 99% percentile
            """

            if x > perc_99:
                return perc_99
            else:
                return x

        for b in range(8):
            perc_99 = np.quantile(self.image_data_3d[:, :, b], 0.99)
            change_max_1_func = np.vectorize(change_max_1)
            self.image_data_3d[:, :, b] = change_max_1_func(self.image_data_3d[:, :, b])

    def normalize_channels(self, num_channel_end, num_channel_start=0):
        """
        Method for normalizing channels of satellite image data.

        Parameters
        ----------
        num_channel_end : int
            The last channel to normalize.
        num_channel_start : int, optional
            The first channel to normalize. Defaults to 0.

        """
        for b in range(num_channel_start, num_channel_end):
            min_val_b = np.min(self.image_data_3d[:, :, b])
            max_val_b = np.max(self.image_data_3d[:, :, b])
            self.image_data_3d[:, :, b] = (self.image_data_3d[:, :, b] - min_val_b) / (
                max_val_b - min_val_b
            )

    def compute_indices(self):
        """
        Method for computing spectral indices from satellite image data.

        It computes 3 indices: SAVI (Soil Adjusted Vegetation Index), NDWI (Normalized Difference Water Index) and NDBI (Normalized Difference Built-up Index).
        These indices are computed from the red, green, blue, near infrared, and short-wave infrared channels of the satellite image data.

        The computed indices are stored in the image_data_3d attribute of the ImageDataset object (3d numpy array) in the 8th, 9th, and 10th channels.

        """
        savi = spyndex.computeIndex(
            index=["SAVI"],
            params={
                "L": 0.25,
                "R": self.image_data_3d[:, :, 2],
                "N": self.image_data_3d[:, :, 3],
            },
        )
        ndwi = spyndex.computeIndex(
            index=["NDWI"],
            params={"G": self.image_data_3d[:, :, 1], "N": self.image_data_3d[:, :, 3]},
        )
        ndbi = spyndex.computeIndex(
            index=["NDBI"],
            params={
                "S1": self.image_data_3d[:, :, 4],
                "N": self.image_data_3d[:, :, 3],
            },
        )

        self.image_data_3d[:, :, 8] = savi
        self.image_data_3d[:, :, 9] = ndwi
        self.image_data_3d[:, :, 10] = ndbi

    def create_dataset(self, summer_image_path, winter_image_path):
        """
        Method for creating image dataset (image_data_3d attribute).

        Parameters
        ----------
        summer_image_path : str
            path of the summer satellite image
        winter_image_path : str
            path of the winter satellite image

        """

        self.add_channels(summer_image_path, winter_image_path)
        self.saturation_contrast()
        self.normalize_channels(num_channel_end=8)
        self.compute_indices()
        self.normalize_channels(8, 11)

    def make_image_2d(self):
        """
        Method for creating 2D numpy array of size (height*width, number of channels) from 3D numpy array of size (height, width, number of channels).

        It reshapes 3D numpy array to 2D numpy array and normalizes the values of the 2D array to be between 0 and 1.

        The resulting 2D numpy array is stored in the image_data_2d_arr attribute of the ImageDataset object.

        """
        new_shape = (
            self.image_data_3d.shape[0] * self.image_data_3d.shape[1],
            self.image_data_3d.shape[2],
        )
        img_as_2d_arr = self.image_data_3d[:, :, :].reshape(new_shape)
        scaler = MinMaxScaler()
        img_as_2d_arr = scaler.fit_transform(img_as_2d_arr)
        self.image_data_2d_arr = img_as_2d_arr


def rasterize_polygons(shapefile_path, raster_path, rasterized_polygons_path):
    """
    Rasterize the given shapefile to the given raster, and write the result to the given path.

    Parameters
    ----------
    shapefile_path : str
        path of the shapefile to be rasterized
    raster_path : str
        path of the raster to be used as the output size and projection
    rasterized_polygons_path : str
        path to write the rasterized result to

    Returns
    -------
    roi : 2D numpy array
        2D numpy array of size (height, width) containing ROI class labels
    """

    vector_polygons = ogr.Open(shapefile_path)
    vector_layer = vector_polygons.GetLayerByIndex(0)
    raster_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()
    raster_ds = None

    memory_driver = gdal.GetDriverByName("GTiff")
    out_raster_ds = memory_driver.Create(
        rasterized_polygons_path, ncol, nrow, 1, gdal.GDT_Byte
    )
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)
    b = out_raster_ds.GetRasterBand(1)
    b.Fill(0)

    gdal.RasterizeLayer(
        out_raster_ds,
        [1],
        vector_layer,
        None,
        None,
        [0],
        ["ALL_TOUCHED=TRUE", "ATTRIBUTE=class_id"],
    )
    out_raster_ds = None

    roi_ds = gdal.Open(rasterized_polygons_path, gdal.GA_ReadOnly)
    roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)

    return roi


def make_feature_matrix(roi, image_dataset):
    """
    Create a feature matrix from the given ROI and image dataset.

    Parameters
    ----------
    roi : 2D numpy array
        2D numpy array of size (height, width) containing ROI class labels
    image_dataset : 3D numpy array
        3D numpy array of size (height, width, number of channels) containing satellite image data

    Returns
    -------
    data : pandas DataFrame
        pandas DataFrame containing the feature matrix
    """
    X = image_dataset[roi > 0, :]
    y = roi[roi > 0]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    data = {
        "blue_summer": X[:, 0],
        "green_summer": X[:, 1],
        "red_summer": X[:, 2],
        "nir_summer": X[:, 3],
        "swir_summer": X[:, 4],
        "blue_winter": X[:, 5],
        "green_winter": X[:, 6],
        "red_winter": X[:, 7],
        "savi": X[:, 8],
        "ndwi": X[:, 9],
        "ndbi": X[:, 10],
    }
    data = pd.DataFrame(data)
    data["class"] = y

    return data


class LandClassModel:
    '''
    A class used to represent a Land Classification Model for satellite images.
    Attributes
    model_path : str
        The file path to a pre-trained model to be loaded.
        A model object (e.g., from scikit-learn) to be used for satellite image land classification.
    model_params : dict
        Parameters of the model.
    model_metrics : dict
        A dictionary containing model metrics such as accuracy, kappa score, etc.
    Methods
    compute_metrics(lst_y_pred, y_test)
    test_model(X_train, y_train, X_test, y_test)
    create_confusion_matrix(y_test, y_pred)
    predict_for_image(img_2d_array, image_dataset)
    compute_execution_time(img_2d_array, image_dataset, area_num)
    visualize_classification_map(pred_maps, map_title)
    make_model_report(X_train, y_train, X_test, y_test, img_2d_arrays, img_datasets, pred_maps)
    '''
    def __init__(self, model_name, model_metrics=None, model=None, model_path=None):
        """
        Initializes a LandClassModel instance.

        Parameters
        ----------
        model_name : str
            The name of the model.
        model_metrics : dict, optional
            A dictionary containing model metrics such as accuracy, kappa score, etc. Defaults to None.
        model : object, optional
            A model object (e.g., from scikit-learn) to be used for satellite image land classification. Defaults to None.
        model_path : str, optional
            The file path to a pre-trained model to be loaded. If provided, the model will be loaded from this path. Defaults to None.

        """
        self.model_path = model_path
        if self.model_path is None:
            self.model = model
        else:
            self.model = joblib.load(model_path)
        self.model_name = model_name
        self.model_params = self.model.get_params()
        self.model_metrics = model_metrics


    @staticmethod
    def compute_metrics(lst_y_pred, y_test):
        """
        Compute accuracy, kappa score and F1 score for given lists of predicted and true labels.

        Parameters
        ----------
        lst_y_pred : list of array-like
            A list of predicted labels
        y_test : array-like
            True labels

        Returns
        -------
        metrics_dct : dict
            A dictionary containing the computed metrics
        """

        metrics_dct = {
            "Accuracy": [],
            "Kappa score": [],
            "F1 score": [],
        }
        for y_pred in lst_y_pred:
            metrics_dct["Accuracy"].append(round(accuracy_score(y_test, y_pred), 3))
            metrics_dct["Kappa score"].append(
                round(cohen_kappa_score(y_test, y_pred), 3)
            )
            metrics_dct["F1 score"].append(
                round(f1_score(y_test, y_pred, average="macro"), 3)
            )

        return metrics_dct

    def test_model(self, X_train, y_train, X_test, y_test):
        """
        Test the model with the best parameters.

        Parameters
        ----------
        X_train : numpy array
            The training dataset
        y_train : numpy array
            The true labels for the training dataset
        X_test : numpy array
            The test dataset
        y_test : numpy array
            The true labels for the test dataset

        Returns
        -------
        y_pred : numpy array
            The predicted labels for the test dataset

        Notes
        -----
        This function fits the model to the training data, predicts the labels for the
        test dataset, computes the metrics, and prints out them.
        The metrics are also stored in the model_metrics attribute of the
        object.
        """

        self.model = self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        test_metrics_results = self.compute_metrics([y_pred], y_test)
        self.model_metrics = test_metrics_results
        print("Метрики качества классификации:")
        for score_key, score_value in test_metrics_results.items():
            print(f"{score_key}: {score_value[0]}")

        return y_pred

    def create_confusion_matrix(self, y_test, y_pred):
        """
        Create a confusion matrix of true labels vs predicted labels.

        Parameters
        ----------
        y_test : 2D numpy array
            The true labels for the test dataset
        y_pred : 2D numpy array
            The predicted labels for the test dataset

        Notes
        -----
        The confusion matrix is plotted using a 'Blues' colormap, and the
        x-axis labels are the predicted labels, and the y-axis labels are the
        true labels.
        """
        print("Матрица несоответствий:")
        cm = confusion_matrix(y_test, y_pred, labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.model.classes_
        )
        disp.plot(cmap="mako_r")
        plt.grid(alpha=0)
        plt.xlabel("Предсказанные метки")
        plt.ylabel("Истинные метки")
        plt.show()

    def predict_for_image(self, img_2d_array, image_dataset):
        """
        Make a prediction on the given 2D image array using the given model.

        Parameters
        ----------
        model : object
            A model object (e.g. from scikit-learn) to be used for satellite image land classification
        img_2d_array : 2D numpy array
            2D numpy array of size (height*width, number of channels) containing satellite image data
        image_dataset : 3D numpy array
            3D numpy array of size (height, width, number of channels) containing satellite image data

        Returns
        -------
        model_pred : 2D numpy array
            2D numpy array of size (height, width) containing predicted class labels
        """
        model_pred = self.model.predict(img_2d_array)
        model_pred = model_pred.reshape(image_dataset[:, :, 0].shape)

        return model_pred

    def compute_execution_time(self, img_2d_array, image_dataset, area_num):
        """
        Compute the execution time of predicting the class labels for a given 2D image array in seconds using exponential notation.

        Parameters
        ----------
        img_2d_array : 2D numpy array
            2D numpy array of size (height*width, number of channels) containing satellite image data
        image_dataset : 3D numpy array
            3D numpy array of size (height, width, number of channels) containing satellite image data
        area_num : int
            The number (№) of the area to compute the execution time for

        """
        start_time = time.process_time()
        self.predict_for_image(img_2d_array, image_dataset)
        end_time = time.process_time()
        execution_time = end_time - start_time
        exp_exec_time = "{:e}".format(execution_time)
        print(f"Время выполнения для участка № {area_num}: {exp_exec_time} секунд")

    def visualize_classification_map(self, pred_maps, map_title):
        """
        Visualize classification maps.

        Parameters
        ----------
        pred_maps : list of 2D numpy arrays
            A list of 2D numpy arrays of size (height, width) containing predicted class labels of each experimental areas
        map_title : str
            The title of the classification map

        Notes
        -----
        The class labels are visualized using a color map, with the class labels on the color bar
        labeled as 'Водные объекты', 'Леса', 'Редколесья', 'Луга', 'Антропогенные объекты'.
        The color map is defined globally as `class_map`.
        """

        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(20, 6), dpi=200)
        axes = axes.ravel()
        for i, pred_map in zip(list(range(3)), pred_maps):
            scalebar = AnchoredSizeBar(
                axes[i].transData,
                100,
                "1 км",
                "lower right",
                sep=6,
                pad=0.5,
                borderpad=0.5,
                color="black",
                bbox_transform=axes[i].transAxes,
                size_vertical=2,
            )
            image_plot = axes[i].imshow(pred_map, cmap=class_map, interpolation="none")
            axes[i].axis("off")
            axes[i].set_title(f"Участок № {i + 1}")
            scalebar.set_clip_on(False)
            axes[i].add_artist(scalebar)
        ep.draw_legend(image_plot, titles=classes)
        plt.suptitle(map_title, fontsize=20, fontweight=600)
        fig.tight_layout()

    def make_model_report(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        img_2d_arrays,
        img_datasets,
        pred_maps,
    ):
        """
        Create a report about the model.

        Parameters
        ----------
        X_train : numpy array
            The training features
        y_train : numpy array
            The training labels
        X_test : numpy array
            The test features
        y_test : numpy array
            The test labels
        img_2d_arrays : list of 2D numpy arrays
            A list of 2D numpy arrays of size (height*width, number of channels) containing satellite image data
        img_datasets : list of 3D numpy arrays
            A list of 3D numpy arrays of size (height, width, number of channels) containing satellite image data
        pred_maps : list of 2D numpy arrays
            A list of 2D numpy arrays of size (height, width) containing predicted class labels of each experimental areas

        Notes
        -----
        This function prints out the results of the model, including
        the confusion matrix, the average execution time of each experimental area, and the visualization of
        the classification maps for each experimental area.
        """
        print(f"Отчет о модели: {self.model_name}")
        print("-----------------------------------------------------------------")
        y_pred = self.test_model(X_train, y_train, X_test, y_test)
        print("-----------------------------------------------------------------")
        self.create_confusion_matrix(y_test, y_pred)
        print("-----------------------------------------------------------------")
        print("Среднее время выполнения:")
        for img_arr, image_dataset, area_num in zip(
            img_2d_arrays, img_datasets, ["1", "2", "3"]
        ):
            model_pred = self.predict_for_image(img_arr, image_dataset)
            self.compute_execution_time(img_arr, image_dataset, area_num)
            pred_maps.append(model_pred)
        print("--------------------------------------------------------")
        self.visualize_classification_map(
            pred_maps,
            f"""Карты классификации территории экспериментальных участков. Метод создания - {self.model_name}""",
        )


class ClearingsExtractor:
    """
    A class representing an extractor for power line clearings from satellite imagery.

    Attributes:
      None

    Methods:
      __init__(self): Initializes the ClearingsExtractor instance.
      edge_detector(self, pred_map): Detects edges of bare lands in a given land classification map.
      open_basemap(self, image_basemap_path): Opens a basemap image from a given path and returns it as a 3D numpy array of size (height, width, 3) with dtype uint8.
      find_lines(self, bare_lands_edges): Finds power line clearings in a given edge image.
      extract(self, pred_map, summer_image_path): Finds power line clearings on a given classification map.

    Notes:
      The `edge_detector` method detects the edges using Gaussian smoothing, setting a threshold for binarization of the resulting image and finding the difference between the binarization result and its binary erosion.
      The `find_lines` method finds power line clearings in an edge image using the probabilistic Hough transform.
    """

    def edge_detector(self, pred_map):
        """
        Detects edges of bare lands in a given land classification map.

        Parameters:
        ----------
        pred_map : 2D numpy array
            A 2D numpy array of size (height, width) containing predicted class labels.

        Returns:
        -------
        model_bare_lands : 2D numpy array
            A 2D numpy array of size (height, width) containing binary mask of bare lands (Class №4) from land classification map.
        binary_smoothed_bare_lands : 2D numpy array
            A 2D numpy array of size (height, width) containing binary mask of smoothed bare lands by gaussian filter.
        edges : 2D numpy array
            A 2D numpy array of size (height, width) containing binary mask of edges of bare lands.
        """
        model_bare_lands = pred_map == 4
        bare_lands_without_noise = binary_opening(model_bare_lands, disk(3))
        smoothed_bare_lands = gaussian_filter(
            bare_lands_without_noise.astype(float), sigma=5
        )
        binary_smoothed_bare_lands = smoothed_bare_lands > 0.5

        eroded_bare_lands = binary_erosion(binary_smoothed_bare_lands, disk(1))
        edges = binary_smoothed_bare_lands.astype(int) - eroded_bare_lands.astype(int)

        return model_bare_lands, binary_smoothed_bare_lands, edges


    def find_lines(self, bare_lands_edges):
        """
        Finds power line clearings in a given edge image.

        Parameters
        ----------
        bare_lands_edges : 2D numpy array
            A 2D numpy array of size (height, width) containing the edges of Class №4 ('Луга') after applying Canny edge detection

        Returns
        -------
        power_line_clearings : 3D numpy array
            A 3D numpy array of size (height, width, 4) RGBA image containing the detected power line clearings as yellow lines of width 3
        """
        lines = probabilistic_hough_line(
            image=bare_lands_edges, threshold=65, line_length=50, line_gap=50
        )
        power_line_clearings = Image.new(
            "RGBA", (bare_lands_edges.shape[1], bare_lands_edges.shape[0]), (0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(power_line_clearings)
        for line_coordinates in lines:
            draw.line(xy=line_coordinates, fill="red", width=3)
        power_line_clearings = np.array(power_line_clearings)

        return power_line_clearings

    def extract(self, pred_map):
        """
        Finds power line clearings on a given classification map.

        Parameters
        ----------
        pred_map : 2D numpy array
            A 2D numpy array of size (height, width) containing predicted class labels

        Returns
        -------
        power_line_clearings : 3D numpy array
            A 3D numpy array of size (height, width, 4) RGBA image containing the detected power line clearings as yellow lines of width 3
        """

        bare_lands_edges = self.edge_detector(pred_map)[-1]
        power_line_clearings = self.find_lines(bare_lands_edges)

        return power_line_clearings

    def visualize_algorithm_steps(self, pred_map, image):
        """
        Visualizes the steps of the power line clearings detection algorithm.

        Parameters:
        pred_map (2D numpy array): A 2D numpy array of size (height, width) containing predicted class labels.
        image (3D numpy array): A 3D numpy array of size (height, width, 3) containing the satellite image.

        Returns:
        fig (matplotlib Figure): A matplotlib Figure object containing the visualized algorithm steps.
        """
        letters = ["а", "б", "в", "г"]

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 11))
        ax = ax.ravel()
        model_bare_lands, bare_lands_without_noise, bare_lands_edges = (
            self.edge_detector(pred_map)
        )
        ax[0].imshow(model_bare_lands, cmap="mako")
        ax[1].imshow(bare_lands_without_noise, cmap="mako")
        ax[2].imshow(bare_lands_edges, cmap="mako", interpolation="none")
        ep.plot_rgb(image, rgb=[2, 1, 0], ax=ax[3], stretch=True)
        power_line_corridors = self.find_lines(bare_lands_edges)
        ax[3].imshow(power_line_corridors)
        for col in range(0, 4):
            ax[col].axis("off")
        plt.tight_layout()

        return fig



def find_clearing_algorithm(summer_image_path, winter_image_path, model):
    """
    Final algorithm that finds power line clearings in a given land classification map.

    Parameters
    ----------
    model : sklearn model
        Trained machine learning model for land classification
    summer_image_path : str
        Path of the summer satellite image
    winter_image_path : str
        Path of the winter satellite image

    Returns
    -------
    power_line_clearings : 3D numpy array
        A 3D numpy array of size (height, width, 4) RGBA image containing the detected power line clearings as yellow lines of width 3
    """

    image_dataset = ImageDataset()
    image_dataset.create_dataset(summer_image_path, winter_image_path)
    image_dataset.make_image_2d()
    pred_map = model.predict_for_image(
        image_dataset.image_data_2d_arr, image_dataset.image_data_3d
    )
    clearing_extractor = ClearingsExtractor()
    power_line_clearings = clearing_extractor.extract(pred_map)

    return power_line_clearings


class ClearingsDetectionMetrics:
    """
    A class representing metrics for power line clearings detection.

    Attributes
    ----------
    power_line_clearings_mask : 2D numpy array
        A 2D numpy array of size (height, width) containing binary mask of power line clearings
    locations : list of tuples
        A list of tuples of (x, y) coordinates of manually selected locations of power line clearings
    closeness_value : float
        A value representing the closeness of detected power line clearings to the manually selected locations
    integrity_value : float
        A value representing the integrity of detected power line clearings
    """

    def __init__(
        self,
        power_line_clearings_mask,
        locations=None,
        closeness_value=None,
        integrity_value=None,
    ):
        """
        __init__ method for ClearingsDetectionMetrics class.

        Parameters
        ----------
        power_line_clearings_mask : 2D numpy array
            A 2D numpy array of size (height, width) containing binary mask of power line clearings
        locations : list of tuples, optional
            A list of tuples of coordinates of power towers in the format (x, y)
        closeness_value : float, optional
            A float value of closeness of power line clearings boundaries to power lines
        integrity_value : float, optional
            A float value of integrity of power line clearings boundaries
        """
        self.power_line_clearings_mask = power_line_clearings_mask
        self.locations = locations
        self.closeness_value = closeness_value
        self.integrity_value = integrity_value

    @staticmethod
    def rasterize_vector(vector_filepath, raster_filepath, output_filepath):
        """
        Rasterizes a vector (shapefile) to a raster (GeoTIFF) given a raster template.

        Parameters
        ----------
        vector_filepath : str
            path to the vector file (shapefile)
        raster_filepath : str
            path to the raster file (GeoTIFF) to use as a template
        output_filepath : str
            path to write the output raster to

        Notes
        -----
        The output raster will have the same projection, geotransform, and size as the input raster.
        The burn value for the rasterization is set to 1, meaning that all pixels in the output raster
        that overlap with the input vector will have the value 1.
        """
        raster_ds = gdal.Open(raster_filepath)
        geotransform = raster_ds.GetGeoTransform()
        projection = raster_ds.GetProjection()
        width = raster_ds.RasterXSize
        height = raster_ds.RasterYSize

        driver = gdal.GetDriverByName("GTiff")
        output_ds = driver.Create(output_filepath, width, height, 1, gdal.GDT_Byte)

        output_ds.SetProjection(projection)
        output_ds.SetGeoTransform(geotransform)

        vector_ds = ogr.Open(vector_filepath)
        vector_layer = vector_ds.GetLayer()

        gdal.RasterizeLayer(output_ds, [1], vector_layer, burn_values=[1])

        vector_ds = None
        raster_ds = None
        output_ds = None

    @staticmethod
    def reduce_power_towers_density(power_towers_shp, reduced_shp):
        """
        Reduces the density of power towers in a given shapefile by grouping them into a regular grid of squares of size `grid_size` meters.

        Parameters
        ----------
        power_towers_shp : str
            path to the shapefile of power towers
        reduced_shp : str
            path to write the reduced shapefile to

        Notes
        -----
        The output shapefile will have the same projection, geotransform, and size as the input shapefile.
        The `grid_size` parameter controls the density of the power towers in the output shapefile.
        A smaller value of `grid_size` will result in a higher density of power towers in the output shapefile.
        A larger value of `grid_size` will result in a lower density of power towers in the output shapefile.
        """
        gdf = gpd.read_file(power_towers_shp)
        gdf.to_crs(epsg=32647, inplace=True)

        minx, miny, maxx, maxy = gdf.total_bounds

        grid_size = 2000
        grid = []

        for x in np.arange(minx, maxx, grid_size):
            for y in np.arange(miny, maxy, grid_size):
                grid.append(
                    Polygon(
                        [
                            (x, y),
                            (x + grid_size, y),
                            (x + grid_size, y + grid_size),
                            (x, y + grid_size),
                        ]
                    )
                )

        grid_gdf = gpd.GeoDataFrame(geometry=grid, crs=gdf.crs)
        gdf["grid_cell"] = gpd.sjoin(gdf, grid_gdf, how="left", op="within")[
            "index_right"
        ]

        reduced_gdf = gdf.groupby("grid_cell").first()
        reduced_gdf.set_crs("EPSG:32647", allow_override=True, inplace=True)
        reduced_gdf.to_file(reduced_shp)

    def open_raster(self, raster_filepath):
        """
        Opens a raster file and extracts the power tower locations.

        Parameters:
        raster_filepath (str): The path to the raster file.

        Returns:
        None. The function sets the 'locations' attribute of the object to a 2D numpy array containing the power tower locations.
        The array is of size (height, width) and contains 1's representing power tower locations and 0's representing background.
        """
        raster = rxr.open_rasterio(raster_filepath)
        data = np.zeros((raster.shape[1], raster.shape[2]), dtype=np.uint8)
        data = np.array(raster[0, :, :])
        self.locations = data


    def split_points_into_arrays(self):
        """
        Splits a 2D numpy array of power tower locations into individual arrays,
        each containing a single power tower location.

        Parameters
        ----------
        self.locations : 2D numpy array
            2D numpy array of size (height, width) containing power tower locations as 1's on a background of 0's

        Returns
        -------
        point_arrays : list
            A list of 2D numpy arrays of size (height, width), each containing a single power tower location
        """
        indices = np.argwhere(self.locations == 1)

        point_arrays = []

        for index in indices:
            one_point_array = np.zeros_like(self.locations)
            one_point_array[tuple(index)] = 1
            point_arrays.append(one_point_array)

        return point_arrays

    @staticmethod
    def calculate_distance_matrix(locations_array):
        """
        Calculate the distance matrix for a given array of location of power lines or power tower.

        Parameters
        ----------
        locations_array : 2D numpy array
            A 2D numpy array of size (height, width) containing locations of interest as 1's on a background of 0's

        Returns
        -------
        distance_matrix : 2D numpy array
            A 2D numpy array of the same size as `locations_array` representing the distance from each background pixel (0's in `locations_array`) to the nearest location of interest (1's in `locations_array`)
        """

        background_mask = locations_array == 0
        distance_matrix = distance_transform_edt(background_mask)

        return distance_matrix

    def mask_distances(self, distance_matrix):
        """
        Masks the distances to power line clearings boundaries by multiplying the distance matrix with the power line clearings mask.

        Parameters
        ----------
        distance_matrix : 2D numpy array
            A 2D numpy array of the same size as `locations_array` representing the distance from each background pixel (0's in `locations_array`) to the nearest location of interest (1's in `locations_array`)

        Returns
        -------
        masked_zeros_distances : 2D numpy masked array
            A 2D numpy masked array of the same size as `distance_matrix` with 0's replaced by np.nan
        """

        masked_distances = distance_matrix * self.power_line_clearings_mask
        masked_zeros_distances = np.ma.masked_equal(masked_distances, 0)

        return masked_zeros_distances

    def calculate_closeness_value(self, raster_filepath):
        """
        Calculates the closeness value from a given raster file of power lines location.

        Parameters
        ----------
        raster_filepath : str
            path of the raster file

        Notes
        -----
        The closeness value is calculated as the reciprocal of the median distance to the power line clearings stands in meters.
        The distances are calculated using the Euclidean distance transform.
        The median distance is calculated after masking the distances to the power line clearings boundaries.
        The calculated closeness value is stored in the `closeness_value` attribute of the object.
        """
        self.open_raster(raster_filepath)
        distance_matrix = self.calculate_distance_matrix(self.locations)
        masked_zeros_distances = self.mask_distances(distance_matrix)
        median_distance = np.ma.median(masked_zeros_distances)
        self.closeness_value = round(1 / median_distance, 3)

    def calculate_integrity_value(self, raster_filepath):
        """
        Calculates the integrity value from a given raster file of power tower locations.

        Parameters
        ----------
        raster_filepath : str
            path of the raster file

        Notes
        -----
        The integrity value is calculated as the reciprocal of the mean minimum distance to the power line clearings stands in meters.
        The distances are calculated using the Euclidean distance transform.
        The minimum distance is calculated after masking the distances to the power line clearings boundaries.
        The calculated integrity value is stored in the `integrity_value` attribute of the object.
        """
        self.open_raster(raster_filepath)
        points_arrays = self.split_points_into_arrays()

        sum_min_mask_dist = 0
        n_towers = len(points_arrays)
        for tower in points_arrays:
            distance_matrix = self.calculate_distance_matrix(tower)
            masked_zeros_distances = self.mask_distances(distance_matrix)
            min_mask_dist = masked_zeros_distances.min()
            sum_min_mask_dist += min_mask_dist
        integrity_value = sum_min_mask_dist / n_towers
        self.integrity_value = round(1 / integrity_value, 3)
