# FEATOUT

### Motivation:

Convolutional Neural Networks (CNNs) have found extensive application in recent years. While they are very successful on large benchmark datasets, training on small and noisy datasets is still challenging. A particular problem is so-called “shortcut learning” [1], where models only learn to classify data based on a very limited set of features; just sufficient to achieve high performance on the train data, but not enough to generalize to unseen data. A common strategy to tackle this issue is data augmentation, e.g., cropping or rotating images randomly to make the data more diverse [2, 3]. Here, we suggest a more targeted strategy: Instead of randomly changing the image, the dataset should be modified based on the model’s attention. Consider the following example: a model should learn to distinguish dogs and cats, and it has learnt to distinguish dogs just by the shape of their ears (shortcut learning). To “encourage” the model to learn other features as well, one could blur or cut the dogs’ ears out of the images. In the next epoch, where the input are the modified dog images, the network hopefully learns to distinguish dogs based on other features, e.g. from the tail. Since specific features are removed in this training scheme, we call this idea “Featout”. A schematic representation of the idea is shown below.

![Featout](assets/featout.png)

### Task:
Implement a simple version of a “Featout” pipeline and demonstrate it on an image dataset! The basic idea is to modify the training dataset based on the model’s attention. You are free to choose 1) the dataset, 2) the model architecture, 3) how to derive the model’s attention, and 4) how to modify the input data. Some hints on each of these points:
* 1) Dataset: Think about when the Featout method could be most successful. In particular, it might not make sense to use a super large dataset that already covers very diverse samples for each class. Also, you could try to demonstrate that Featout helps to generalize to different data (make the test data significantly different from the training data, e.g. different light / blur)
* 2) Model: Probably, any architecture should be fine, but some model architectures support interpretability
* 3) Model attention: Note that the model’s attention must be derived *efficiently*. Since the dataset should be modified during training, a method with too much computational effort is not feasible. Thus, gradient-based methods are most suitable (this repo uses a simple gradient saliency method, see [here](featout/interpret.py))
* 4) Feature removal: The idea is to prevent the model from learning the same features in the next epoch. It is up to you how to do this. In this repo, the parts of the image with most attention are either blurred or masked with zeros (see [here](featout/utils/blur.py))


### Deliverables:
Submit a link to your GitHub repository, as well as a graphical abstract of your solution. The Readme should describe your solution.

### Getting started: 
For inspiration, we provide a codebase with a simple implementation of Featout in this repository. In the folder “papers”, you can find some related work. The one that is most related to the Featout idea is the paper by Wang et al [4] proposing a method called CamDrop.

### References

[1] [Geirhos, Robert, et al. "Shortcut learning in deep neural networks." Nature Machine Intelligence 2.11 (2020): 665-673.](papers/geirhos_2020.pdf)

[2] [DeVries, Terrance, and Graham W. Taylor. "Improved regularization of convolutional neural networks with cutout." arXiv preprint arXiv:1708.04552 (2017).](papers/DeVries_2017.pdf)

[3] [Zhong, Zhun, et al. "Random erasing data augmentation." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 07. 2020.](papers/zhong_2017.pdf)

[4] [Wang, Hongjun, et al. "Camdrop: A new explanation of dropout and a guided regularization method for deep neural networks." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.](papers/wang_2019.pdf)