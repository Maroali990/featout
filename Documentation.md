# Lalas: Quick Experimentation with Featout Method

## Overview

This document records a brief experimental investigation we conducted over a few hours to evaluate the featout method's influence on image classification tasks. Due to resource constraints and time limitations, our exploration was brief and primarily focused on the QuickDraw and Food-101 datasets. It is important to note that this was a quick and limited-scope experiment; the findings are initial observations. We edited the requirements.txt, so make sure to install the missing dependencies. For all the experiments with featout, we used featout in every epoch except the first one.

## Experimental Setup

### Dataset 1

[QuickDraw Dataset](https://quickdraw.withgoogle.com/data)


#### Classes
- Cookies
- Brain

#### Data Split
- Test: 200
- Validation: 800

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

**Remarks**:
After spending considerable time on the initial classes, we decided to pivot our focus to another dataset to gather more insights in our limited time frame.

### Dataset 3: 
[Food-101 Dataset](https://www.kaggle.com/dansbecker/food-101)
[To download](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)


#### Classes
- Greek Salad
- Beet Salad

#### Techniques Tested
- Normal Blurring
- Feature Inverting with and without featout

#### Epochs
- 5

#### Results

| Method                                 | Accuracy (%) |
|----------------------------------------|--------------|
| Normal Blurring                        | No Improvement |
| Feature Inverting with featOut         | 76           |
| Feature Inverting without featOut      | 72           |

**Remarks**:
We did not notice any significant performance improvement with the normal blurring method. However, a slight enhancement was observed when we employed feature inverting coupled with the featout method. This document was created with the assistance of ChatGPT.

