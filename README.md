# Pytorch_Transfer-learning
Let us first explore this model's layers and then make a decision as to which ones we want to freeze. By freeze we mean that we want the parameters of those layers to be fixed. When fine tuning a model, we are basically taking a model trained on Dataset A, and then training it on a new Dataset B. We could potentially start the training from scratch as well, but it would be like re-inventing the wheel. Let me explain why.

Suppose, I want to train a dataset to learn to differentiate between a car and a bicycle. Now, I could potentially gather images of both categories and train a network from scratch. But, given the majority of work already out there, it's easy to find a model trained to identify things like Dogs, cats, and humans. Admittedly, neither of these 3 look like cars or bicycles. However, it's still better than nothing. We could start by taking this model, and train it to learn car v/s bicycle. Gains : 1) It will be faster, 2) We need lesser images of cats and bicycles.


Now, let's take a look at the contents of a resnet18.
We use the function .children() for this purpose. This lets us look at the contents/layers of a model. 
Then, we use the .parameters() function to access the parameters/weights of any layer. 
Finally, every parameter has a property .requires_grad which defines whether a parameter is trained or frozen. By default it is True, and the network updates it in every iteration. If it is set to False, then it is not updated and is said to be "frozen".


Important Note
Now that you have frozen this network, another thing changes to make this work. That is your optimizer. Your optimizer is the one which actually updates these values. By default, the models are written like this -

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)

But, this will give you an error as this will try to update all the parameters of model. However, you've set a bunch of them to frozen! So, the way to pass only the ones still being updated is -

optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)

SECTION 2 - Model Saving/Loading
There's 2 primary ways in which models are saved in PyTorch. The suggested one is using "state dictionaries". They're faster and requires lower space. Basically, they have no idea of the model structure, they're just the values of the parameters/weights. So, you must create your model with the required architecture and then load the values. The architecture is declared as we did it above.
SECTION 3 - changing last layer, deleting last layer, adding layers¶
ADDING LAYERS¶

Say, you want to add a fully connected layer to the model we have right now. One obvious way would be to edit the list I discussed above and appending to it another layer. However, often times we have such a model trained and want to see if we can load that model, and add just a new layer on top of it. As mentioned above, the loaded model should have the SAME architecture as saved one, and so we can't use the list method.

We need to add layers on top. The way to do this is simple in PyTorch - We just need to create a custom model! And this brings us to our next section - creating custom models!

SECTION 4 - CUSTOM MODELS : Combining sections 1-3 and adding layers on top
Let's make a custom model. As mentioned above, we will load half of the model from a pre-trained network. This seems complicated, right? Half the model is trained, half is new. Further, we want some of it to be frozen. Some to be update-able. Really, once you've done this, you can do anything with model architectures in PyTorch.


SECTION 5 - CUSTOM LOSS FUNCTIONS
Now that we have our model all in place we can load anything and create any architecture we want. That leaves us with 2 important components in any pipeline - Loading the data, and the training part. Let's take a look at the training part. The two most important components of this step are the optimizer and the loss function. The loss function quantifies how far our existing model is from where we want to be, and the optimizer decides how to update parameters such that we can minimize the loss.

Sometimes, we need to define our own loss functions. And here are a few things to know about this -

custom Loss functions are defined using a custom class too. They inherit from torch.nn.Module just like the custom model.
Often, we need to change the dimenions of one of our inputs. This can be done using view() function.
If we want to add a dimension to a tensor, use the unsqueeze() function.
The value finally being returned by a loss function MUST BE a scalar value. Not a vector/tensor.
The value being returned must be a Variable. This is so that it can be used to update the parameters. The best way to do so is to just make sure that both x and y being passed in are Variables. That way any function of the two will also be a Variable.
A Pytorch Variable is just a Pytorch Tensor, but Pytorch is tracking the operations being done on it so that it can backpropagate to get the gradient.
Here I show a custom loss called Regress_Loss which takes as input 2 kinds of input x and y. Then it reshapes x to be similar to y and finally returns the loss by calculating L2 difference between reshaped x and y. This is a standard thing you'll run across very often in training networks.

Consider x to be shape (5,10) and y to be shape (5,5,10). So, we need to add a dimension to x, then repeat it along the added dimension to match the dimension of y. Then, (x-y) will be the shape (5,5,10). We will have to add over all three dimensions i.e. three torch.sum() to get a scalar.
