# DeepFish-dataset
We focus accurate localization of fish in addition to the total length of the fish.
## Model Architecture
  * This model is an extension of [Retina-NET](https://paperswithcode.com/method/retinanet)
  * The regression head is modified to be able to predict the total length of the fish.
      * We introduced a new loss function to minimize the total relative error of the output.
          
