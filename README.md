This is an example of deployment fastai model in c++ application. 

The goal is to show how to get the same predictions from fastai model with python and from c++ app with LibTorch. 
The meaning of prediction is not important, so I use a random image and do not train a model.
In order to transfer the model to c++ I used Torch Script representation.

The prediction from fastai Learner:

![python output](https://i.imgur.com/adHDOCZ.png)

The prediction from c++ app:

![cpp output](https://i.imgur.com/AYAVvlB.png)
