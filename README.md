# optical-character-recognition
A simple implementation of an optical character recognition problem using SVM. The main goal of this project is to recognize chacarters of lisence plates from a given database.

# Related work
This project is a simplified implementation of an OCR architecture proposed by [Gonçalves et al. (2016)](/ITSC-2016.pdf), which proposes a solution to recognize license plates in real-time using temporal redundancy.
![Architecture](/arch.png)
<p align="center">Sequence of tasks performed by the proposed approach (<a href="/ITSC-2016.pdf">Gonçalves et al., 2016</a>).</p>

# Database
The database used is private, so it's not possible to provide the files in this repository. However, all you need to know about the database used in this project is:

#### Images and notes
Each image have a related text file, which describes the bounding boxes related to the lisence plate recognized in the image and each character of the plate. The note file also have the real value of the characters of the lisence plate. Example:
```python
text: XXX-9999
position_plate: 568 672 99 37
position_chars:
	char0: 573 687 12 18
	char1: 585 687 12 17
	char2: 597 687 12 18
	char3: 614 687 12 17
	char4: 627 687 12 17
	char5: 639 687 11 17
	char6: 651 687 12 17
```
#### Directory structure
The database was divided in three sets: training, test and validation. The images was grouped by folders. These grouped images represent a video clip, which each image represents a frame of the video. These group of images will be used to simulate the temporal redundancy solution.
```
database
├ training
| ├ Track1
| | ├ Track1[01].png
| | ├ Track1[01].txt
| | ├ ...
| | ├ Track1[M].png
| | └ Track1[M].txt
| ├ ...
| └ TrackN
|   └ ...
├ test
| └ ...
└ validation
  └ ...
```

# Technologies used
This project was made using [Python](https://www.python.org/) language. The libraries used are:
* [Scikit-learn](http://scikit-learn.org/) to get a SVM implementation;
* [Scikit-image](http://scikit-image.org/) to get a HOG describer implementation;
* [OpenCV](https://www.opencv.org/) to read and handle images;
* [NumPy](http://www.numpy.org) to make some numeric transformations necessary for SVM input;
* [Matplotlib](https://matplotlib.org/) to plot some graphs in order to analyse the results.

# Development
This project is a simplified implementation of the OCR architecture proposed by [Gonçalves et al. (2016)](/ITSC-2016.pdf); more particularly, related to the **character recognition** and **temporal redundancy aggregation** steps. The information given for the used database allows us to jump the steps related to *vehicle detection*, *lisence plate detection* and *characters segmentation*.

**Support Vector Machines (SVM)** was the model used to predict the character values. I've also used the **Radial Basis Function (RBF)** kernel, which is the State-of-Art kernel to OCR problems. To describe the images, I've used the **Histogram of Oriented Gradients (HOG)** describer.

To work with multiple classes, I've used the **One-against-all** composition. To do so, one SVM is created to each classes of the problem (in this case, the letters \[a to z] and the numbers \[0 to 9]). On the training step, these SVMs receive items from classes 1 or 0, where 1 means that the given item is from the same class that this SVM is responsible for, and 0 otherwise. On the forecasting, the input image is provided to all SVMs, and the SVM with the highest answer value has the chosen class.

About the temporal redundancy agregation, the same lisence plate is recognized multiple times. The final value is given by a voting process from these multiple results.

# Results
I've reached a precision of **99,7%**, using this approach and the given database as input. To simply describe the experiment, 5523 characters (789 images) was used on the training step, and 5628 characters (804 images) was used on the test step. Although, 5613 images was predicted correctly, against 15 wrong predictions. The image below describes the confusion matrix got from the experiment.

![Confusion matrix](/confusion.png)
