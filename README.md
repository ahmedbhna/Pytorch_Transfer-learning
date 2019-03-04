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
