# Lalas: Quick Experimentation with Featout Method

## Overview

This document records a brief experimental investigation we conducted over a few hours to evaluate the featout method's influence on image classification tasks. Due to resource constraints and time limitations, our exploration was brief and primarily focused on the QuickDraw and Food-101 datasets. It is important to note that this was a quick and limited-scope experiment; the findings are initial observations. We edited the requirements.txt, so make sure to install the missing dependencies. 
For all the experiments with featout, we used featout in every epoch except the first one. We tried out several image modification methods to find the best way to mitigate shortcut learning, such as blurring, inverting the features, and texture shuffling. You can find the script [Here](https://github.com/Maroali990/featout/blob/master/featout/utils/blur.py). 

## Experimental Setup

### Data Split
- 80/20

### Dataset 1

[QuickDraw Dataset](https://quickdraw.withgoogle.com/data)


#### Classes
- Cookies
- Brain
#### Number of images: 100, 1000, 10000


#### Epochs
- 10

#### Results

| Method           | Accuracy (%) |
|------------------|--------------|
| With featOut     | 90           |
| Without featOut  | 90           |

**Remarks**:
The model quickly adapted to the classification task, showing high accuracy within just 10 epochs, both with and without applying the featout method.

### Dataset 2: 
[QuickDraw Dataset](https://quickdraw.withgoogle.com/data)

#### Classes
- Cats
- Faces 
#### Number of images: 100, 1000, 10000


**Remarks**:
After spending considerable time on the initial classes, we focused on another dataset to gather more insights in our limited time frame.

### Dataset 3: 
[Food-101 Dataset](https://www.kaggle.com/dansbecker/food-101)
[To download](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)


#### Classes
- Greek Salad
- Beet Salad
#### Number of images: 1000


#### Techniques Tested
- Normal Blurring with featout
- Feature Inverting with featout
- Texture shuffle with featout

#### Epochs
- 5

#### Results

| Method                                 | Accuracy (%) |
|----------------------------------------|--------------|
| Without featout                        | 72           |
| Normal Blurring with featout           | 72           |
| Feature Inverting with featout         | 76           |
| Texture Shuffle with featout           | 76           |

**Remarks**:
We did not notice any significant performance improvement with the normal blurring method. However, a slight enhancement was observed when we employed feature inverting or texture shuffle coupled with the featout method. This document was created with the assistance of ChatGPT.

