import time
from osgeo import ogr, gdal
import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import earthpy.plot as ep
import spyndex
from PIL import Image, ImageDraw
import skimage
from skimage.morphology import binary_opening
from skimage.util import img_as_ubyte
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

classes = ['Водные объекты',
           'Леса',
           'Редколесья',
           'Луга',
           'Антропогенные объекты']
colors = ['#0070BB', '#00A550', '#55DD33', '#FCF55F', '#E44D2E']
class_map = ListedColormap(colors)

class ImageDataset:

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


  def add_channels(self, url_summer_image, url_winter_image):

    """
    Method for adding channels from summer and winter images to image dataset.

    Parameters
    ----------
    url_summer_image : str
        URL of the summer satellite image
    url_winter_image : str
        URL of the winter satellite image
    """
    
    summer_image = rxr.open_rasterio(url_summer_image)
    winter_image = rxr.open_rasterio(url_winter_image)

    image_dataset = np.zeros((summer_image.shape[1], summer_image.shape[2], 11),
                              dtype = np.float64)

    for b in range(5):
        image_dataset[:, :, b] = summer_image[b, :, :]
    for b in range(5, 8):
        image_dataset[:, :, b] = winter_image[b-5, :, :]

    self.image_data_3d = image_dataset


  def saturation_contrast(self):

    """
    Method for contrast stretching of satellite image channels.
    """
    for b in range(8):
      perc_99 = np.quantile(self.image_data_3d[:, :, b], 0.99)
      change_max_1 = lambda x: perc_99 if x > perc_99 else x
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
      self.image_data_3d[:, :, b] = (self.image_data_3d[:, :, b] - min_val_b) / (max_val_b - min_val_b)


  def compute_indices(self):

    """
    Method for computing spectralindices from satellite image data.

    It computes 3 indices: SAVI (Soil Adjusted Vegetation Index), NDWI (Normalized Difference Water Index) and NDBI (Normalized Difference Built-up Index).
    These indices are computed from the red, green, blue, near infrared, and short-wave infrared channels of the satellite image data.

    The computed indices are stored in the image_data_3d attribute of the ImageDataset object (3d numpy array) in the 8th, 9th, and 10th channels.

    """
    savi = spyndex.computeIndex(index = ['SAVI'],
                                params = {'L': 0.25,
                                        'R': self.image_data_3d[:, :, 2],
                                        'N': self.image_data_3d[:, :, 3]})
    ndwi = spyndex.computeIndex(index=['NDWI'],
                                params={'G': self.image_data_3d[:, :, 1],
                                        'N': self.image_data_3d[:, :, 3]})
    ndbi = spyndex.computeIndex(index = ["NDBI"],
                                params = {'S1': self.image_data_3d[:, :, 4],
                                        'N': self.image_data_3d[:, :, 3]})

    self.image_data_3d[:, :, 8] = savi
    self.image_data_3d[:, :, 9] = ndwi
    self.image_data_3d[:, :, 10] = ndbi


  def create_dataset(self, url_summer_image, url_winter_image):

    """
    Method for creating image dataset (image_data_3d attribute).

    Parameters
    ----------
    url_summer_image : str
        URL of the summer satellite image
    url_winter_image : str
        URL of the winter satellite image

    """
    
    self.add_channels(url_summer_image, url_winter_image)
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
    new_shape = (self.image_data_3d.shape[0] * self.image_data_3d.shape[1],
                 self.image_data_3d.shape[2])
    img_as_2d_arr = self.image_data_3d[:, :, :].reshape(new_shape)
    scaler = MinMaxScaler()
    img_as_2d_arr = scaler.fit_transform(img_as_2d_arr)
    self.image_data_2d_arr = img_as_2d_arr

  
def rasterize_polygons(url_shapefile, url_raster, url_rasterized_polygons):

  """
  Rasterize the given shapefile to the given raster, and write the result to the given URL.

  Parameters
  ----------
  url_shapefile : str
      URL of the shapefile to be rasterized
  url_raster : str
      URL of the raster to be used as the output size and projection
  url_rasterized_polygons : str
      URL to write the rasterized result to
      
  Returns
  -------   
  ROI : 2D numpy array
      2D numpy array of size (height, width) containing ROI class labels
  """
  
  vector_polygons = ogr.Open(url_shapefile)
  vector_layer = vector_polygons.GetLayerByIndex(0)
  raster_ds = gdal.Open(url_raster, gdal.GA_ReadOnly)
  ncol = raster_ds.RasterXSize
  nrow = raster_ds.RasterYSize
  proj = raster_ds.GetProjectionRef()
  ext = raster_ds.GetGeoTransform()
  raster_ds = None

  memory_driver = gdal.GetDriverByName('GTiff')
  out_raster_ds = memory_driver.Create(url_rasterized_polygons,
                                       ncol, nrow, 1, gdal.GDT_Byte)
  out_raster_ds.SetProjection(proj)
  out_raster_ds.SetGeoTransform(ext)
  b = out_raster_ds.GetRasterBand(1)
  b.Fill(0)

  status = gdal.RasterizeLayer(out_raster_ds, [1],
                               vector_layer, None, None, [0],
                               ['ALL_TOUCHED=TRUE', 'ATTRIBUTE=class_id'])
  out_raster_ds = None

  roi_ds = gdal.Open(url_rasterized_polygons, gdal.GA_ReadOnly)
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
    'blue_summer' : X[:, 0],
    'green_summer': X[:, 1],
    'red_summer': X[:, 2],
    'nir_summer': X[:, 3],
    'swir_summer': X[:, 4],
    'blue_winter': X[:, 5],
    'green_winter': X[:, 6],
    'red_winter': X[:, 7],
    'savi': X[:, 8],
    'ndwi': X[:, 9],
    'ndbi': X[:, 10]
    }
  data = pd.DataFrame(data)
  data['class'] = y

  return data


class LandClassificationModel:

  def __init__(self, model, model_name, model_params=None, model_metrics=None):
    """
    Parameters
    ----------
    model : object
        A model object (e.g. from scikit-learn) to be used for satellite image land classification
    model_name : str
        A string name for the model
    model_params : dict, optional
        A dictionary of model parameters, by default None
    model_metrics : dict, optional
        A dictionary of model metrics (e.g. accuracy, kappa score, etc.), by default None
    """
    
    self.model = model
    self.model_name = model_name
    self.model_params = model_params
    self.model_metrics = model_metrics


  @staticmethod
  def compute_metrics(lst_y_pred, y_test):

    """
    Compute accuracy, kappa score, precision, recall, and F1 score for given lists of predicted and true labels.

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

    metrics_dct = {'Accuracy': [], 'Kappa score': [], 'Precision': [],
                  'Recall': [], 'F1 score': []}
    for y_pred in lst_y_pred:
      metrics_dct['Accuracy'].append(round(accuracy_score(y_test, y_pred), 3))
      metrics_dct['Kappa score'].append(round(cohen_kappa_score(y_test, y_pred), 3))
      metrics_dct['Precision'].append(round(precision_score(y_test, y_pred, average='macro'), 3))
      metrics_dct['Recall'].append(round(recall_score(y_test, y_pred, average='macro'), 3))
      metrics_dct['F1 score'].append(round(f1_score(y_test, y_pred, average='macro'), 3))

    return metrics_dct


  def validate_model_params(self, lst_params, X_train=X_train, y_train=y_train, X_validate=X_validate, y_validate=y_validate):

    """
    Validate a list of model parameters.

    Parameters
    ----------
    lst_params : list of dict
        A list of dictionaries of model parameters

    Notes
    -----
    This function prints out a Pandas DataFrame of the validation metrics
    for each set of model parameters, with columns for the different metrics
    and rows for each set of parameters. The index of the DataFrame is the
    number of the set of parameters (starting from 1). The columns are
    labeled with the name of the metric.
    """
    lst_y_val_pred = []

    for params in lst_params:
      self.model.set_params(**params)
      self.model = self.model.fit(X_train, y_train)
      y_val_pred = self.model.predict(X_validate)
      lst_y_val_pred.append(y_val_pred)

    val_metrics_results = self.compute_metrics(lst_y_val_pred, y_validate)

    model_set_params_num = list(range(1, len(lst_params) + 1))
    val_metrics_results = pd.DataFrame(val_metrics_results,
                                       index=model_set_params_num)
    val_metrics_results.columns.name = 'Метрика'
    val_metrics_results.index.name = '№ набора параметров'
    print(val_metrics_results)


  def test_model(self, best_params, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):

    """
    Test the model with the best parameters.

    Parameters
    ----------
    best_params : dict
        A dictionary of the best parameters (e.g. selected from the validation set) for the model

    Returns
    -------
    y_pred : numpy array
        The predicted labels for the test dataset

    Notes
    -----
    This function sets the model parameters to the best parameters,
    fits the model to the training data, predicts the labels for the
    test dataset, computes the metrics, and prints out them.
    The metrics are also stored in the model_metrics attribute of the
    object.
    """
    self.model.set_params(**best_params)
    self.model_params = best_params
    self.model = self.model.fit(X_train, y_train)
    y_pred = self.model.predict(X_test)

    test_metrics_results = self.compute_metrics([y_pred], y_test)
    self.model_metrics = test_metrics_results
    print('Метрики качества классификации:')
    for score_key, score_value in test_metrics_results.items():
      print(f'{score_key}: {score_value[0]}')

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
    print('Матрица несоответствий:')
    cm = confusion_matrix(y_test, y_pred, labels=self.model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=self.model.classes_)
    disp.plot(cmap='Blues')
    plt.grid(alpha=0)
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.show()


  @staticmethod
  def predict_for_image(model, img_2d_array, image_dataset):

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
    model_pred = model.predict(img_2d_array)
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
    self.predict_for_image(self.model, img_2d_array, image_dataset)
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
    
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(20,6))
    axes = axes.ravel()
    for i, pred_map in zip(list(range(3)), pred_maps):
      scalebar = AnchoredSizeBar(axes[i].transData, 100, '1 км', 'lower right', sep=6,
                              pad=0.5, borderpad=0.5, color='black',
                              bbox_transform=axes[i].transAxes, size_vertical=2)
      image_plot = axes[i].imshow(pred_map, cmap=class_map, interpolation='none')
      axes[i].axis('off')
      axes[i].set_title(f'Участок № {i+1}')
      scalebar.set_clip_on(False)
      axes[i].add_artist(scalebar)
    ep.draw_legend(image_plot, titles=classes)
    plt.suptitle(map_title, fontsize=20, fontweight=600)
    fig.tight_layout()


  def make_model_report(self, lst_params, best_params, y_test, img_2d_arrays, img_datasets, pred_maps):

    """
    Create a report about the model.

    Parameters
    ----------
    lst_params : list of dict
        A list of dictionaries of model parameters
    best_params : dict
        A dictionary of the best model parameters
    img_2d_arrays : list of 2D numpy arrays
        A list of 2D numpy arrays of size (height*width, number of channels) containing satellite image data
    img_datasets : list of 3D numpy arrays
        A list of 3D numpy arrays of size (height, width, number of channels) containing satellite image data
    pred_maps : list of 2D numpy arrays
        A list of 2D numpy arrays of size (height, width) containing predicted class labels of each experimental areas

    Notes
    -----
    This function prints out the results of the validation of the model parameters,
    the confusion matrix, the average execution time of each experimental area, and the visualization of
    the classification maps for each experimental area.
    """
    print(f"Отчет о модели: {self.model_name}")
    print("-----------------------------------------------------------------")
    print('Результаты валидации параметров:')
    self.validate_model_params(lst_params)
    print("-----------------------------------------------------------------")
    y_pred = self.test_model(best_params)
    print("-----------------------------------------------------------------")
    self.create_confusion_matrix(y_test, y_pred)
    print("-----------------------------------------------------------------")
    print('Среднее время выполнения:')
    for img_arr, image_dataset, area_num in zip(img_2d_arrays,
                                                img_datasets,
                                                ['1', '2', '3']):
      model_pred = self.predict_for_image(self.model, img_arr, image_dataset)
      self.compute_execution_time(img_arr, image_dataset, area_num)
      pred_maps.append(model_pred)
    print("--------------------------------------------------------")
    self.visualize_classification_map(pred_maps, f"""Карты классификации территории участков. Метод создания - {self.model_name}""")
    
    
class ClearingsExtractor:

  def edge_detector(self, pred_map):

    """
    Detects edges of bare lands in a given land classification map.

    Parameters
    ----------
    pred_map : 2D numpy array
        A 2D numpy array of size (height, width) containing predicted class labels

    Returns
    -------
    model_bare_lands : 2D numpy array
        A 2D numpy array of the same size as `pred_map` with the class labels of 'Луга' (class №4)
    bare_lands_without_noise : 2D numpy array
        A 2D numpy array of the same size as `pred_map` with the class labels of 'Луга' (class №4) after applying binary opening
    bare_lands_edges : 2D numpy array
        A 2D numpy array of the same size as `pred_map` with the edges of 'Луга' (class №4) after applying Canny edge detection
    """
    model_bare_lands = pred_map == 4
    bare_lands_without_noise = binary_opening(model_bare_lands,
                                      skimage.morphology.disk(3))
    bare_lands_edges = canny(bare_lands_without_noise, sigma=5)

    return model_bare_lands, bare_lands_without_noise, bare_lands_edges

  @staticmethod
  def open_basemap(url_image_basemap):

    """
    Opens a basemap image from a given URL and returns it as a 3D numpy array of size (height, width, 3) with dtype uint8.

    Parameters
    ----------
    url_image_basemap : str
        URL of the basemap image

    Returns
    -------
    rgb_image : 3D numpy array
        3D numpy array of size (height, width, 3) containing the basemap image data with dtype uint8
    """

    image = rxr.open_rasterio(url_image_basemap)
    image = np.asarray(image)
    rgb_image = np.rollaxis(image[::-1][0:3], 0, 3)
    rgb_image = img_as_ubyte(rgb_image)

    return rgb_image


  def find_power_line_clearings(self, bare_lands_edges):

    """
    Finds power line clearings in a given edge image.

    Parameters
    ----------
    bare_lands_edges : 2D numpy array
        A 2D numpy array of size (height, width) containing the edges of Class №4 ('Луга') after applying Canny edge detection

    Returns
    -------
    lines : list
        A list of detected power line clearings as 2D arrays of size (2, 2)
    power_line_clearings : 3D numpy array
        A 3D numpy array of size (height, width, 4) RGBA image containing the detected power line clearings as yellow lines of width 3
    """
    lines = probabilistic_hough_line(image=bare_lands_edges, threshold=100,
                                     line_length=100, line_gap=50)
    power_line_clearings = Image.new('RGBA', (bare_lands_edges.shape[1],
                                              bare_lands_edges.shape[0]),
                                             (0, 0, 0, 0))
    draw = ImageDraw.Draw(power_line_clearings)
    for l in lines:
      draw.line(xy=l, fill='yellow', width=3)
    power_line_clearings = np.array(power_line_clearings)

    return lines, power_line_clearings


  def extract(self, pred_map, url_image_basemap):

    """
    Finds power line clearings on a given classification map.

    Parameters
    ----------
    pred_map : 2D numpy array
        A 2D numpy array of size (height, width) containing predicted class labels
    url_image_basemap : str
        URL of the basemap image

    Returns
    -------
    power_line_clearings : 3D numpy array
        A 3D numpy array of size (height, width, 4) RGBA image containing the detected power line clearings as yellow lines of width 3
    """
    
    bare_lands_edges = self.edge_detector(pred_map)[-1]
    rgb = self.open_basemap(url_image_basemap)
    power_line_clearings = self.find_power_line_clearings(bare_lands_edges)[-1]

    return power_line_clearings


  def visualize_algorithm_steps(self, pred_maps, images_url):

    """
    Visualize the results of the power line clearing extraction algorithm steps.

    Parameters
    ----------
    pred_maps : list of 2D numpy arrays
        A list of 2D numpy arrays of size (height, width) containing predicted class labels
    images_url : list of str
        A list of URLs of the basemap images

    Notes
    -----
    This function visualizes the results of the power line clearing extraction algorithm steps
    for each experimental area. The results are visualized in a figure with 3 rows and 5 columns,
    where each row corresponds to an experimental area and each column corresponds to a step of the algorithm.
    The columns are labeled with the step number and the row is labeled with the experimental area number.
    """
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(21, 13))
    for row_num, pred_map, image_url in zip(range(len(pred_maps)), pred_maps, images_url):
      model_bare_lands, bare_lands_without_noise, bare_lands_edges = self.edge_detector(pred_map)
      ax[row_num, 0].imshow(model_bare_lands)
      ax[row_num, 0].set_title(f"""Маска класса лугов из
      карты классификации, участок № {row_num+1}""")
      ax[row_num, 1].imshow(bare_lands_without_noise)
      ax[row_num, 1].set_title(f"""Маска класса лугов,
      обработанная двоичным открытием, участок № {row_num+1}""")
      ax[row_num, 2].imshow(bare_lands_edges, cmap='Greys_r', interpolation='none')
      ax[row_num, 2].set_title(f"""Результат детектора
      границ Канни, участок № {row_num+1}""")
      rgb = self.open_basemap(image_url)
      lines, power_line_corridors = self.find_power_line_clearings(bare_lands_edges)
      ax[row_num, 3].imshow(rgb*12)
      for line in lines:
        p0, p1 = line
        ax[row_num, 3].plot((p0[0], p1[0]), (p0[1], p1[1]))
      ax[row_num, 3].set_title(f"""Результат вероятностного
      преобразования Хафа, участок № {row_num+1}""")
      ax[row_num, 4].imshow(rgb*12)
      ax[row_num, 4].imshow(power_line_corridors, alpha=0.6)
      ax[row_num, 4].set_title(f"""Результат выделения лесных
    просек, участок № {row_num+1}""")
      for col in range(0, 5):
        ax[row_num, col].axis('off')
    plt.suptitle("Результаты выделения лесных просек под ЛЭП на маске лугов из карты классификации",
                fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    
def find_clearing_algorithm(url_summer_image, url_winter_image):
  """
  Final algorithm that finds power line clearings in a given land classification map.

  Parameters
  ----------
  url_summer_image : str
      URL of the summer satellite image
  url_winter_image : str
      URL of the winter satellite image

  Returns
  -------
  power_line_clearings : 3D numpy array
      A 3D numpy array of size (height, width, 4) RGBA image containing the detected power line clearings as yellow lines of width 3
  """
  
  image_dataset = ImageDataset()
  image_dataset.create_dataset(url_summer_image, url_winter_image)
  image_dataset.make_image_2d()
  pred_map = LandClassificationModel.predict_for_image(logreg.model,
                                                       image_dataset.image_data_2d_arr,
                                                       image_dataset.image_data_3d)
  clearing_extractor = ClearingsExtractor()
  power_line_clearings = clearing_extractor.extract(pred_map, url_summer_image)

  return power_line_clearings