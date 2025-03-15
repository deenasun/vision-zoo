# HW 1 Worksheet

---

This is the worksheet for Homework 1. Your deliverables for this homework are:

- [ ] This worksheet with all answers filled in. If you include plots/images, be sure to include all the required files. Alternatively, you can export it as a PDF and it will be self-sufficient.
- [ ] Kaggle submission and writeup (details below)
- [ ] Github repo with all of your code! You need to either fork it or just copy the code over to your repo. A simple way of doing this is provided below. Include the link to your repo below. If you would like to make the repo private, please dm us and we'll send you the GitHub usernames to add as collaborators.

`https://github.com/deenasun/vision-zoo`

## To move to your own repo:

First follow `README.md` to clone the code. Additionally, create an empty repo on GitHub that you want to use for your code. Then run the following commands:

```bash
$ git remote rename origin staff # staff is now the provided repo
$ git remote add origin <your repos remote url>
$ git push -u origin main
```



# Part -1: PyTorch review

Feel free to ask your NMEP friends if you don't know!

## -1.0 What is the difference between `torch.nn.Module` and `torch.nn.functional`?

`torch.nn.Module is a PyTorch collection that includes components for building neural networks with 4 main classes: Parameters, Containers, Layers, and Functions. The classes and functions in torch.nn automatically manage parameters like weights and biases.`

## -1.1 What is the difference between a Dataset and a DataLoader?

`A Dataset is a collection of sample points and their labels. A Dataloader is a class that wraps an iterable around a Dataset and provides additional functionality such as shuffling the data and batching it.`

## -1.2 What does `@torch.no_grad()` above a function header do?

`We use a `torch.no_grad()` context manager when we want to prevent a computational graph from being built. This is used in situations when we don't need to keep track of the partial derivatives, backpropagate, or update parameters.`

# Part 0: Understanding the codebase

Read through `README.md` and follow the steps to understand how the repo is structured.

## 0.0 What are the `build.py` files? Why do we have them?

`The build.py files are used to set up and build DataLoaders and models using the parameters specified in the config files. The build.py files read in the parameters from the configs then call the classes to build the models/data loaders.`

## 0.1 Where would you define a new model?

`New models are defined in the /models directory.`

## 0.2 How would you add support for a new dataset? What files would you need to change?

`To add a new dataset, we would edit the /data/datasets.py by defining a custom class for a Dataset with an __init__, __len__, and __getitem__ function`

## 0.3 Where is the actual training code?

`The training loop is in main.py.`

## 0.4 Create a diagram explaining the structure of `main.py` and the entire code repo.

Be sure to include the 4 main functions in it (`main`, `train_one_epoch`, `validate`, `evaluate`) and how they interact with each other. Also explain where the other files are used. No need to dive too deep into any part of the code for now, the following parts will do deeper dives into each part of the code. For now, read the code just enough to understand how the pieces come together, not necessarily the specifics. You can use any tool to create the diagram (e.g. just explain it in nice markdown, draw it on paper and take a picture, use draw.io, excalidraw, etc.)

![Alt text](/images/main_diagram.jpeg)


# Part 1: Datasets

The following questions relate to `data/build.py` and `data/datasets.py`.

## 1.0 Builder/General

### 1.0.0 What does `build_loader` do?

`Based on what the dataset defined in config is, build_loader will first get the training, validation, and test splits from the dataset. Then it will create 3 DataLoaders using the training, validation, and test splits.`

### 1.0.1 What functions do you need to implement for a PyTorch Dataset? (hint there are 3)

`To create a custom PyTorch Dataset, we need to implement the functions __init__, __getitem__, and __len__.`

## 1.1 CIFAR10Dataset

### 1.1.0 Go through the constructor. What field actually contains the data? Do we need to download it ahead of time?

`The field "dataset" contains the data. The parameter root="/data/cifar10" indicates the directory that the dataset will be stored in. But since we set download=True, if the dataset doesn't exist in this directory already, the constructor will download it. We don't need to downlaod the data ahead of time because if it doesn't already exist, PyTorch will load it for us.`

### 1.1.1 What is `self.train`? What is `self.transform`?

`In a Dataset, the instance variable self.train defines whether or not the Dataset is a training set or a validation/test set. self.transform is a method of the CIFAR10Dataset class that defines a series of transformations to perform on the data. In addition to normalizing and resizing, for training datasets, each sample point's contrast, brightness, and saturation are randomized and the image is randomly flipped.`

### 1.1.2 What does `__getitem__` do? What is `index`?

`__getitem__ returns the item at index "index" of the dataset along with its corresponding label. "index" is an argument specifying which index of the dataset to retrieve from.`

### 1.1.3 What does `__len__` do?

`__len__ returns the total number of sample points in the dataset.`

### 1.1.4 What does `self._get_transforms` do? Why is there an if statement?

`self._get_transforms returns a series of PyTorch Transforms composed together that are used to process data before it is used for training or testing. The if statement adds different transform layers into the transform sequence depending on if the Dataset will be used for training or not. If the Dataset is used for training, every sample image's contrast, brightness, and saturation are randomly modified and it is flipped with probabilty of 0.5.`

### 1.1.5 What does `transforms.Normalize` do? What do the parameters mean? (hint: take a look here: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)

`transforms.Normalize normalizes a tensor representation of the image. The first parameter is the mean (channel-wise), the second parameter is the standard deviation (channel-wise).`

## 1.2 MediumImagenetHDF5Dataset

### 1.2.0 Go through the constructor. What field actually contains the data? Where is the data actually stored on honeydew? What other files are stored in that folder on honeydew? How large are they?

`The data for the MediumImagenetHDF5 Dataset is stored in the "file" field. Based on the filepath, the data should be stored in the folder /honey/data/nmep in the file medium-imagenet-96.hdf5.`

> *Some background*: HDF5 is a file format that stores data in a hierarchical structure. It is similar to a python dictionary. The files are binary and are generally really efficient to use. Additionally, `h5py.File()` does not actually read the entire file contents into memory. Instead, it only reads the data when you access it (as in `__getitem__`). You can learn more about [hdf5 here](https://portal.hdfgroup.org/display/HDF5/HDF5) and [h5py here](https://www.h5py.org/).

### 1.2.1 How is `_get_transforms` different from the one in CIFAR10Dataset?

`The _get_transforms function in the MediumImagenetHDF5 Dataset also regularizes the values in each sample point to be bewteen 0 to 255 and normalizes each sample point with a mean and standard deviation. For this Dataset, we also apply a resizing that scales the image by 2. The additional steps of randomly flipping the image and randomizing the image's brightness, contrast, and saturation are applied only if the Dataset is a training split and the Dataset's augment parameter is set to True.`

### 1.2.2 How is `__getitem__` different from the one in CIFAR10Dataset? How many data splits do we have now? Is it different from CIFAR10? Do we have labels/annotations for the test set?

`The __getitem__ function for the MediumImagenet Dataset first loads the image file associated with the Dataset's split. If the Dataset isn't a test set, then the function also gets its corresponding label; otherwise, the image's label is set as -1. Finally, we transform the image and cast the label into a tensor. There are 3 data splits: train, val, and test. Only the test set has labels.`

### 1.2.3 Visualizing the dataset

Visualize ~10 or so examples from the dataset. There's many ways to do it - you can make a separate little script that loads the datasets and displays some images, or you can update the existing code to display the images where it's already easy to load them. In either case, you can use use `matplotlib` or `PIL` or `opencv` to display/save the images. Alternatively you can also use `torchvision.utils.make_grid` to display multiple images at once and use `torchvision.utils.save_image` to save the images to disk.

Be sure to also get the class names. You might notice that we don't have them loaded anywhere in the repo - feel free to fix it or just hack it together for now, the class names are in a file in the same folder as the hdf5 dataset.
`Visualizing the images from the imagenet dataset`
![Alt text](/images/visualize_imagenet.png)

`Visualizing the sample images from the imagenet training Dataloader`
![Alt text](/images/visualize_dataloader_train.png)

# Part 2: Models

The following questions relate to `models/build.py` and `models/models.py`.

## What models are implemented for you?

`LeNet and ResNet`

## What do PyTorch models inherit from? What functions do we need to implement for a PyTorch Model? (hint there are 2)

`PyTorch models inherit from nn.Module. A PyTorch model needs to implement the __init__ and forward functions`

## How many layers does our implementation of LeNet have? How many parameters does it have? (hint: to count the number of parameters, you might want to run the code)

`LeNet has 2 convolution layers (each followed by a sigmoid activation function and a pooling layer) plus a classifier with 3 hidden layers. This implementation has 99.28K parameters.`


# Part 3: Training

The following questions relate to `main.py`, and the configs in `configs/`.

## 3.0 What configs have we provided for you? What models and datasets do they train on?

`configs/ has configs for a base LeNet model for CIFAR10, a base ResNet18 model for CIFAR10, and a base ResNet model for Imagenet.`

## 3.1 Open `main.py` and go through `main()`. In bullet points, explain what the function does.

```* Builds the loader specified by the config
* Builds the model specified by the config
* Displays information about the model's parameters and flop counts
* Builds an optimizer for the model
* If the config specifies that we are resuming training for a model, load the checkpoint
* Otherwise begin training the model from scratch:
    * For each epoch:
        * Train one epoch of the model
        * Log the model's training accuracy and training loss
        * Validate the model and log its valiation accuracy and validation loss
        * Save a checkpoint
        * Print the max accuracy so far during training
* Log information about the total training time
* Evaluate the model on the test DataLoader```

## 3.2 Go through `validate()` and `evaluate()`. What do they do? How are they different? 
> Could we have done better by reusing code? Yes. Yes we could have but we didn't... sorry...

`validate computes the outputs of the model on the validation set then calculates the accuracy of the model by comparing the model's predictions with the true labels of the validation set. evaluate() simply computes the outputs of the model given a dataset without comparing the model's predictions with anything.`


# Part 4: AlexNet

## Implement AlexNet. Feel free to use the provided LeNet as a template. For convenience, here are the parameters for AlexNet:

```
Input NxNx3 # For CIFAR 10, you can set img_size to 70
Conv 11x11, 64 filters, stride 4, padding 2
MaxPool 3x3, stride 2
Conv 5x5, 192 filters, padding 2
MaxPool 3x3, stride 2
Conv 3x3, 384 filters, padding 1
Conv 3x3, 256 filters, padding 1
Conv 3x3, 256 filters, padding 1
MaxPool 3x3, stride 2
nn.AdaptiveAvgPool2d((6, 6)) # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
flatten into a vector of length x # what is x?
Dropout 0.5
Linear with 4096 output units
Dropout 0.5
Linear with 4096 output units
Linear with num_classes output units
```

> ReLU activation after every Conv and Linear layer. DO **NOT** Forget to add activations after every layer. Do not apply activation after the last layer.

## 4.1 How many parameters does AlexNet have? How does it compare to LeNet? With the same batch size, how much memory do LeNet and AlexNet take up while training? 
> (hint: use `gpustat`)

`AlexNet has 57.82324M parameters while LeNet has 99.28K.

AlexNet takes up 1456 MB while training while LeNet takes up 272 MB while training.`

## 4.2 Train AlexNet on CIFAR10. What accuracy do you get?

Report training and validation accuracy on AlexNet and LeNet. Report hyperparameters for both models (learning rate, batch size, optimizer, etc.). We get ~77% validation with AlexNet.

> You can just copy the config file, don't need to write it all out again.
> Also no need to tune the models much, you'll do it in the next part.

```AlexNet hyperparameters
* Image size: 70
* Learning rate: 0.0003
* Batch size: 256
* Optimizer: AdamW
* Optimizer beta: 0.9
* Optimizer epsilon: 1e-8
* Drop-out rate: 0.5
* Learning rate scheduler: cosine

AlexNet training accuracy after 50 epochs: 99.9%
AlexNet validation accuracy after 50 epochs: 76.79%

LeNet hyperparameters:
* Image size: 32
* Learning rate: 0.0003
* Batch size: 256
* Optimizer: AdamW
* Optimizer beta: 0.9
* Optimizer epsilon: 1e-8
* Drop-out rate: 0.5
* Learning rate scheduler: cosine

LeNet training accuracy after 50 epochs: 37.2%
LeNet validation accuracy after 50 epochs: 38.6%
```

# Part 5: Weights and Biases

> Parts 5 and 6 are independent. Feel free to attempt them in any order you want.

> Background on W&B. W&B is a tool for tracking experiments. You can set up experiments and track metrics, hyperparameters, and even images. It's really neat and we highly recommend it. You can learn more about it [here](https://wandb.ai/site).
> 
> For this HW you have to use W&B. The next couple parts should be fairly easy if you setup logging for configs (hyperparameters) and for loss/accuracy. For a quick tutorial on how to use it, check out [this quickstart](https://docs.wandb.ai/quickstart). We will also cover it at HW party at some point this week if you need help.

## 5.0 Setup plotting for training and validation accuracy and loss curves. Plot a point every epoch.

```
    # initialize wandb run
    run = wandb.init(
        project='vision-zoo',
        name=config.MODEL.NAME,
        config={
            "dataset": config.DATA.DATASET,
            "batch_size": config.DATA.BATCH_SIZE,
            "image_size": config.DATA.IMG_SIZE,
            "learning_rate": config.TRAIN.LR,
            "epochs": config.TRAIN.EPOCHS,
            "optimizer": config.TRAIN.OPTIMIZER.NAME,
        },
    )

    # define number of epochs as custom x-axis
    wandb.define_metric("epoch")
    wandb.define_metric("train_acc", step_metric="epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_acc", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")

    ...

    # saving stats to wandb
        wandb.log({
            "epoch": epoch,
            "train_acc": train_acc1, 
            "train_loss": train_loss, 
            "val_acc": val_acc1, 
            "val_loss": val_loss
        })
```

## 5.1 Plot the training and validation accuracy and loss curves for AlexNet and LeNet. Attach the plot and any observations you have below.

![Alt text](/images/alexnet_vs_lenet.png)
`AlexNet's validation loss starts lower and decreases, but then trends upwards after about 20 epochs whereas LeNet's validation loss decreaes steadily downward. However, AlexNet's training loss converges near 0 whereas LeNet's training loss plateaus around 1.7 even after 50 epohcs. AlexNet reached much higher training and validation accuracies (nearly double or triple that of LeNet).`

## 5.2 For just AlexNet, vary the learning rate by factors of 3ish or 10 (ie if it's 3e-4 also try 1e-4, 1e-3, 3e-3, etc) and plot all the loss plots on the same graph. What do you observe? What is the best learning rate? Try at least 4 different learning rates.

![Alt text](/images/alexnet_learningrates.png)
![Alt text](/images/alexnet_learningrates_runtimes.png)
```
Learning rates tried
(Original) LR: 3e-4    Max accuracy: 77.42%    Final valiation loss: 1.798
LR: 1e-4    Max accuracy: 76.44%    Final validation loss: 1.478
LR: 3e-3    Max accuracy: 10.00%    Final valiation loss: 2.303
LR: 1e-3    Max accuracy: 10.00%    Final valiation loss: 2.303
LR: 1e-5    Max accuracy: 62.11%    Final valiation loss: 1.071

When the learning rates were too high (e.g. 0.003), the model did not learn the data very well. Training accuracies and validation accuracies were very low for the AlexNet model with a learning rate of 0.003. Furthermore, that model had the highest loss and its validation and training losses did not fluctuate at all over the course of 50 epochs.

With the smallest learning rate I tested (1-e5), the training accuracies and validation accuracies were also very low. However, unlike the other models which seemed to start overfitting after about 25 epochs (based on the increasing validation loss), the AlexNet model with a learning rate of 1e-5 was still exhibiting decreasing validation losses. This could be due to the fact that it hadn't yet found a minimum yet for its loss function given how small the step sizes were.
```

## 5.3 Do the same with batch size, keeping learning rate and everything else fixed. Ideally the batch size should be a power of 2, but try some odd batch sizes as well. What do you observe? Record training times and loss/accuracy plots for each batch size (should be easy with W&B). Try at least 4 different batch sizes.

![Alt text](/images/alexnet_batchsizes.png)
![Alt text](/images/alexnet_batchsizes_runtimes.png)
```
Batch sizes tried
(Original) Batch size: 256     Max accuracy: 77.42%%    Final validation loss: 1.798    Total training time: 6 min, 57 sec
Batch size: 128             Max accuracy: 77.47%    Final valiation loss: 1.966     Total training time: 12 min, 35 sec
Batch size: 2048            Max accuracy: 72.72%    Final valiation loss: 0.8826    Total training time: 11 min, 31 sec
Batch size: 5000            Max accuracy: 60.33%    Final valiation loss: 1.105     Total training time: 11 min, 28 sec
Batch size: 23              Max accuracy: 76.33    Final valiation loss: 2.007      Total training time: 29 min, 35 sec
```

## 5.4 As a followup to the previous question, we're going to explore the effect of batch size on _throughput_, which is the number of images/sec that our model can process. You can find this by taking the batch size and dividing by the time per epoch. Plot the throughput for batch sizes of powers of 2, i.e. 1, 2, 4, ..., until you reach CUDA OOM. What is the largest batch size you can support? What trends do you observe, and why might this be the case?
You only need to observe the training for ~ 5 epochs to average out the noise in training times; don't train to completion for this question! We're only asking about the time taken. If you're curious for a more in-depth explanation, feel free to read [this intro](https://horace.io/brrr_intro.html).

![Alt text](/images/alexnet_throughput.png)

`For batch sizes of 1 and 2, it took an extremely long time for the model to train on even just 5 epochs. The model that took the fastest time to train used a batch size of 512. The total training time did not decrease montonically; the total training time also started increasing as the batch sizes got very large, although not to the scale of a batch size of 1. Throughput increased as batch size increased. I reached CUDA OOM with batch_size of 32768---the largest batch size the server could support was 16384.`

## 5.5 Try different data augmentations. Take a look [here](https://pytorch.org/vision/stable/transforms.html) for torchvision augmentations. Try at least 2 new augmentation schemes. Record loss/accuracy curves and best accuracies on validation/train set.

![Alt text](/images/alexnet_transforms.png.png)
```
Base transforms: 76.78% validation accuracy, 99.862% training accuracy
Random crop: 74.20% validation accuracy, 100% training accuracy
Gaussian blur: 73.62% validation accuracy, 100% training accuracy
```

## 5.6 (optional) Play around with more hyperparameters. I recommend playing around with the optimizer (Adam, SGD, RMSProp, etc), learning rate scheduler (constant, StepLR, ReduceLROnPlateau, etc), weight decay, dropout, activation functions (ReLU, Leaky ReLU, GELU, Swish, etc), etc.

`YOUR ANSWER HERE`



# Part 6: ResNet

## 6.0 Implement and train ResNet18

In `models/*`, we provided some skelly/guiding comments to implement ResNet. Implement it and train it on CIFAR10. Report training and validation curves, hyperparameters, best validation accuracy, and training time as compared to AlexNet. 

![Alt text](/images/alexnet_vs_resnet.png)
```
AlexNet
Hyperparameters: learning rate = 0.0003, optimizer = AdamW, epochs = 50
Training time: 6 min 57 secs
Best validation accuracy: 77.26%
Validation loss at epoch 50: 1.79779

ResNet
Hyperparameters: learning rate = 0.0003, optimizer = AdamW, epochs = 50
Training time: 17 min 40 sec
Best validation accuracy: 88.64%
Validation loss at epoch 50: 0.56221

ResNet18 Base
Hyperparameters: learning rate = 0.0003, optimizer = AdamW, epochs = 100
Training time: 31 min 46 sec
Best validation accuracy: 85.92%
Validation loss at epoch 100: 0.701
```

## 6.1 (optional) Visualize examples

Visualize a couple of the predictions on the validation set (20 or so). Be sure to include the ground truth label and the predicted label. You can use `wandb.log()` to log images or also just save them to disc any way you think is easy.

`A few validation images from ResNet trained on the CIFAR10 dataset`
![Alt text](/images/resnet_visualizations.png)

# Part 7: Kaggle submission

To make this more fun, we have scraped an entire new dataset for you! 🎉

We called it MediumImageNet. It contains 1.5M training images, and 190k images for validation and test each. There are 200 classes distributed approximately evenly. The images are available in 224x224 and 96x96 in hdf5 files. The test set labels are not provided :). 

The dataset is downloaded onto honeydew at `/data/medium-imagenet`. Feel free to play around with the files and learn more about the dataset.

For the kaggle competition, you need to train on the 1.5M training images and submit predictions on the 190k test images. You may validate on the validation set but you may not use is as a training set to get better accuracy (aka don't backprop on it). The test set labels are not provided. You can submit up to 10 times a day (hint: that's a lot).

Your Kaggle scores should approximately match your validation scores. If they do not, something is wrong.

(Soon) when you run the training script, it will output a file called `submission.csv`. This is the file you need to submit to Kaggle. You're required to submit at least once. 

## Kaggle writeup

We don't expect anything fancy here. Just a brief summary of what you did, what worked, what didn't, and what you learned. If you want to include any plots, feel free to do so. That's brownie points. Feel free to write it below or attach it in a separate file.

**REQUIREMENT**: Everyone in your group must be able to explain what you did! Even if one person carries (I know, it happens) everyone must still be able to explain what's going on!

Now go play with the models and have some competitive fun! 🎉

```
Some optimizations my group tried:
- Swapped empty nn.Sequential to nn.Identity for the shortcut layer in ResNet
- Wrapped training under torch.amp.autcast and used a Gradscaler for backward passes (AMP = automatic mixed precision, using different data types optimized for different operations)
- Adding persistent_workers=True into the training Dataloader to keep worker processes up between epochs
- Added a prefetch_factor for the Dataloaders
- Tried out different learning rate schedulers, including ReduceLROnPlateau
- Tried out different learning rates
- Tried out different batch sizes

Main challenges:
- Reducing time taken per epoch for ResNet on larger dataset
- Overcoming plateaus in validation accuracy
```