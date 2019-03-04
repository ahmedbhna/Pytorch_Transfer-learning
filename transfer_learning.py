import torch.nn as nn
import torch
from torch.autograd.variable import Variable
from torchvision import datasets, models, transforms
model = models.resnet18(pretrained = False)

child_counter = 0
for child in model.children():
    print(" child", child_counter, "is:")
    print(child)
    child_counter += 1
   
for child in model.children():
    for param in child.parameters():
        print(param)
        break
    break
     
     
child_counter = 0
for child in model.children():
    if child_counter < 6:
        print("child ",child_counter," was frozen")
        for param in child.parameters():
            param.requires_grad = False
    elif child_counter == 6:
        children_of_child_counter = 0
        for children_of_child in child.children():
            if children_of_child_counter < 1:
                for param in children_of_child.parameters():
	                  param.requires_grad = False
	              print('child ', children_of_child_counter, 'of child',child_counter,' was frozen')
            else:
                print('child ', children_of_child_counter, 'of child',child_counter,' was not frozen')
            children_of_child_counter += 1
     else:
         print("child ",child_counter," was not frozen")
      child_counter += 1
#model saving and selectiing RMSprop optimizer
optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
torch.save(model.state_dict(), MODEL_PATH)
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint)


#changing last layer
model = models.resnet18(pretrained = False)
num_final_in = model.fc.in_features
NUM_CLASSES = 300
model.fc = nn.Linear(num_final_in, NUM_CLASSES)
model.fc = nn.Linear(num_final_in, NUM_CLASSES)

#taking the features from the last hidden layer
new_model = nn.Sequential(*list(model.children())[:-1]
new_model_removing_2 = nn.Sequential(*list(model.children())[:-2]


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd.variable import Variable
from torchvision import datasets, models, transforms

# New models are defined as classes. 
class Resnet_Added_Layers_Half_Frozen(nn.Module):
    def __init__(self,LOAD_VIS_URL=None):
        super(ResnetCombinedFull2, self).__init__()
    
        model = models.resnet18(pretrained = False)
        num_final_in = model.fc.in_features
        model.fc = nn.Linear(num_final_in, 300)
        
        # Now that the architecture is defined same as above, let's load the model we would have trained above. 
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint)
        
        
        # Let's freeze the same as above. Same code as above without the print statements
        child_counter = 0
        for child in model.children():
            if child_counter < 6:
                for param in child.parameters():
                    param.requires_grad = False
            elif child_counter == 6:
                children_of_child_counter = 0
                for children_of_child in child.children():
                    if children_of_child_counter < 1:
                        for param in children_of_child.parameters():
                            param.requires_grad = False
                    else:
                    children_of_child_counter += 1

            else:
                print("child ",child_counter," was not frozen")
            child_counter += 1
        
        # Now, let's define new layers that we want to add on top. 
        # Basically, these are just objects we define here. The "adding on top" is defined by the forward()
        
        # NOTE - Even the above model needs to be passed to self.
        self.vismodel = nn.Sequential(*list(model.children()))
        self.projective = nn.Linear(512,400)
        self.nonlinearity = nn.ReLU(inplace=True)
        self.projective2 = nn.Linear(400,300)
        
    
    # The forward function defines the flow of the input data and thus decides which layer/chunk goes on top of what.
    def forward(self,x):
        x = self.vismodel(x)
        x = torch.squeeze(x)
        x = self.projective(x)
        x = self.nonlinearity(x)
        x = self.projective2(x)
        return x
      
class Regress_Loss(torch.nn.Module):
    
    def __init__(self):
        super(Regress_Loss,self).__init__()
        
    def forward(self,x,y):
        y_shape = y.size()[1]
        x_added_dim = x.unsqueeze(1)
        x_stacked_along_dimension1 = x_added_dim.repeat(1,NUM_WORDS,1)
        diff = torch.sum((y - x_stacked_along_dimension1)**2,2)
        totloss = torch.sum(torch.sum(torch.sum(diff)))
        return totloss
