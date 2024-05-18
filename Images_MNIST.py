# -*- coding: utf-8 -*-
"""MNIST-Playground

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1x78L_B1B8tZqFLscX4XCBVaIvgBJWF_p

# Initialization: Imports & Time Management
"""

import time

from keras.datasets import mnist
from keras.utils import to_categorical

import colorsys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

import plotly.io as pio
pio.renderers.default = 'colab'

def time_since_start(start_time):
    current_time = time.time()
    elapsed_time = current_time - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    milliseconds = int((elapsed_time % 1) * 1000)

    return f"{hours}h {minutes}m {seconds}s {milliseconds}ms"

"""# Initialization: Dataset"""

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the training data
x_train = x_train.astype('float32') / 255.0
y_train = to_categorical(y_train)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# Preprocess the testing data
x_test = x_test.astype('float32') / 255.0
y_test = to_categorical(y_test)

# Preprocess the evaluation data - make sure it the model has never seen the examples before
#x_eval = x_train[train_samples:(train_samples+eval_samples)]
#y_eval = y_train[train_samples:(train_samples+eval_samples)]
x_eval = x_test
y_eval = y_test
x_eval = torch.from_numpy(x_eval)
y_eval = torch.from_numpy(y_eval)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

trainSetMNIST = [(torch.flatten(x), torch.flatten(y)) for x, y in zip(x_train, y_train)]
testSetMNIST = [(torch.flatten(x), torch.flatten(y)) for x, y in zip(x_test, y_test)]

print(len(trainSetMNIST), len(testSetMNIST))

"""# Initialization: Data"""

class CustomMNISTData(Dataset):
    def __init__(self, mode="", transform = None):
        if mode=="Train":
            self.input = x_train
            self.output = y_train
        elif mode=="Test":
            self.input = x_test
            self.output = y_test
        elif mode=="Evaluate":
            self.input = x_eval
            self.output = y_eval

        self.transform = transform

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        image = self.input[idx]
        output = self.output[idx]

        return torch.flatten(image), output

"""#Initialization: Network"""

def getLayer(hidden_layers, layer, input_size=10, output_size=10):
    if(hidden_layers[layer][0] == "Linear"):
        return nn.Linear(input_size, output_size)
    elif(hidden_layers[layer][0] == "Conv2d"):
        return nn.Conv2d(input_size, output_size)
    return False

def getActivation(hidden_layers, layer):
    if(hidden_layers[layer][2] == "ReLU"):
        return nn.ReLU()
    elif(hidden_layers[layer][2] == "Sigmoid"):
        return nn.Sigmoid()
    elif(hidden_layers[layer][2] == "Tanh"):
        return nn.Tanh()
    return False

def checkIfActivationLayerExists(hidden_layers, layer):
    if hidden_layers[layer][2] != "None":
        return True
    return False

layers = []
currentLayer = 0
relevantLayerIndices = []
def createLayers(layerName, layerType, activationLayerType):
    global currentLayer

    layers.append((layerName, layerType, activationLayerType))
    relevantLayerIndices.append(currentLayer*2)
    if(activationLayerType != "None"):
        relevantLayerIndices.append((currentLayer*2)+1)
    currentLayer += 1
    return layerName, layerType

class CustomizableRENN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(CustomizableRENN, self).__init__()
        #Add input and output layer to the hidden_layers
        self.num_layers = (len(hidden_sizes))
        self.hidden_layers = hidden_layers

        for layer in range(self.num_layers):
            if layer == 0:

                setattr(self, *createLayers(f'fc{layer}', getLayer(self.hidden_layers, layer, input_size, hidden_layers[layer][1]), self.hidden_layers[layer][2]))
            elif layer == (self.num_layers - 1):
                setattr(self, *createLayers(f'fc{layer}', getLayer(self.hidden_layers, layer, hidden_layers[layer-1][1], output_size), self.hidden_layers[layer][2]))
            else:
                #print(layer, layer-1, self.num_layers, hidden_sizes[layer-1], hidden_sizes[layer])
                setattr(self, *createLayers(f'fc{layer}', getLayer(self.hidden_layers, layer, hidden_layers[layer-1][1], hidden_layers[layer][1]), self.hidden_layers[layer][2]))

            if checkIfActivationLayerExists(self.hidden_layers, layer):
                setattr(self, f'activation{layer}', getActivation(self.hidden_layers, layer))

    def forward(self, x):
        for layer in range(self.num_layers):
            x = getattr(self, f'fc{layer}')(x)

            #There can not be an activation layer at the input and output!
            if (checkIfActivationLayerExists(self.hidden_layers, layer)):
                x = getattr(self, f'activation{layer}')(x)
        return x

#print(len(torch.flatten(torch.tensor(train_dataloader.dataset[0][0]))), len(torch.flatten(torch.tensor(train_dataloader.dataset[0][1]))))
input_size = len(torch.flatten(torch.tensor(train_dataloader.dataset[0][0])))
output_size = len(torch.flatten(torch.tensor(train_dataloader.dataset[0][1])))

model = CustomizableRENN(input_size, hidden_sizes, output_size)
model.to(device)
layers = np.array(layers)

if(loss_function == "MSE"):
    criterion_class = nn.MSELoss()  # For regression
elif(loss_function == "Cross-Entropy"):
    criterion_class = nn.CrossEntropyLoss()  # For multi-class classification

if(optimizer == "Adam"):
    chosen_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif(optimizer == "SGD"):
    chosen_optimizer = optim.SGD(model.parameters(), lr=learning_rate)

"""# Training"""

# Training Loop
def train_model(model, criterion_class,  optimizer, train_dataloader, test_dataloader, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        #print(len(train_dataloader.dataset[0][0]))
        for images, classification in train_dataloader:
            images = images.float()
            images = images.to(device)
            classification = classification.float()
            classification = classification.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion_class(output, classification)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, classification in test_dataloader:
                images = images.float()
                images = images.to(device)
                classification = classification.float()
                classification = classification.to(device)
                output = model(images)
                loss = criterion_class(output, classification)
                val_loss += loss.item()

        # Print statistics
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_dataloader)}, Validation Loss: {val_loss/len(test_dataloader)}')

# Train the model
trainTime = time.time()
train_model(model, criterion_class, chosen_optimizer, train_dataloader, test_dataloader, epochs=epochs)
print(f"Time passed since start of training: {time_since_start(trainTime)}")

"""# Evaluation: Hooks"""

layerSizes = [size[1] for size in hidden_sizes[:]]
activationsBySources = np.full((train_samples, len(layers)*2, np.max(layerSizes)))
activationsByLayers = np.full((len(layers)*2, np.max(layerSizes), train_samples), )
print(activationsBySources.shape)
print(activationsByLayers.shape)

#Variables for usage within the hook
layer = 0
source = 0
dictionaryForSourceLayerNeuron = activationsBySources
dictionaryForLayerNeuronSource = activationsByLayers
train_dataloader = DataLoader(trainSubset, batch_size=1, shuffle=False)
print(layers)

# Forward hook
def forward_hook(module, input, output):
    global layer
    global source
    global dictionaryForSourceLayerNeuron
    global dictionaryForLayerNeuronSource
    global hidden_sizes

    activation_type = type(getActivation(hidden_sizes, int(layer/2)))
    layer_type = type(getLayer(hidden_sizes, int(layer/2)))
    relevantOutput = output[0].cpu().numpy()

    if (type(module) == activation_type or type(module) == layer_type):
        #Use for array structure like: [source, layer, neuron]
        dictionaryForSourceLayerNeuron[source][layer,:layers[int(layer/2)][1].out_features] = relevantOutput

        #Use for array structure like: [layer, neuron, source]
        for neuronNumber, neuron in enumerate(relevantOutput):
            if neuronNumber < layers[int(layer/2)][1].out_features:
                dictionaryForLayerNeuronSource[layer][neuronNumber][source] = neuron
            else:
                break

    if(layer % 2 == 0):
        if(checkIfActivationLayerExists(hidden_sizes, int(layer/2))):
            layer += 1
        elif(layer == (len(layers)*2)-2):
            layer = 0
        else:
            layer += 2
    else:
        if(layer == (len(layers)*2)-1):
            layer = 0
        else:
            layer += 1
#-------------------------------------------------------------------------------

def attachHooks(hookLoader):
    global source

    hooks = []  # Store the handles for each hook
    outputs = np.array([])

    for name, module in model.named_modules():
      if not isinstance(module, CustomizableRENN):
          hook = module.register_forward_hook(forward_hook)
          hooks.append(hook)

    with torch.no_grad():
      # Forward Pass
      convertTime = time.time()
      for source, (inputs, labels) in enumerate(hookLoader):
        # Uncomment for array structure like: [source, layer, neuron]
        inputs = inputs.float()
        inputs = inputs.to(device)
        _ = model(inputs)

      print(f"Time passed since conversion: {time_since_start(convertTime)}")

    # Remove hooks after use
    for hook in hooks:
        hook.remove()

attachHooks(train_dataloader)
activationsBySources = dictionaryForSourceLayerNeuron
activationsByLayers = dictionaryForLayerNeuronSource

activationsByLayers[0][0]

"""# Evaluation: Output"""

import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

def showImagesUnweighted(originalImage, blendedSourceImageActivation, blendedSourceImageSum, closestMostUsedSourceImagesActivation, closestMostUsedSourceImagesSum):
    fig, axes = plt.subplots(1, 5, figsize=(35, 35))
    plt.subplots_adjust(hspace=0.5)

    # Display original image
    axes[0].set_title(f"BLENDED - Original: {originalImage[1]}")
    axes[0].imshow(Image.fromarray(originalImage[0].cpu().numpy()*255))

    # Display blendedSourceImageActivation
    axes[1].set_title(f"A=Activation - Closest Sources/Neuron (Most Used)")
    axes[1].imshow(blendedSourceImageActivation[0])

    # Display blendedSourceImageSum
    axes[2].set_title(f"S=Sum - Closest Sources/Neuron (Most Used)")
    axes[2].imshow(blendedSourceImageSum[0])

    # Display weightedSourceImageActivation
    axes[3].set_title(f"WA=WeightedActivation - Closest Sources/Neuron (Weighted)")
    axes[3].imshow(Image.fromarray(np.zeros(shape=[28,28], dtype=np.uint8)).convert("RGBA"))

    # Display weightedSourceImageSum
    axes[4].set_title(f"WS=WeigthedSum - Closest Sources/Neuron (Weighted)")
    axes[4].imshow(Image.fromarray(np.zeros(shape=[28,28], dtype=np.uint8)).convert("RGBA"))

    plt.show()

    # Display closestMostUsedSourceImagesActivation
    fig, axes = plt.subplots(1, len(closestMostUsedSourceImagesActivation)+2, figsize=(35, 35))
    axes[0].set_title(f"NON-LINEAR - Original: {originalImage[1]}")
    axes[0].imshow(Image.fromarray(originalImage[0].cpu().numpy()*255))
    axes[1].set_title(f"A - Closest Sources/Neuron (Most Used)")
    axes[1].imshow(blendedSourceImageActivation[0])
    for i, source in enumerate(closestMostUsedSourceImagesActivation):
        image = createImageWithPrediction(x_train[source[0]], y_train[source[0]], predict(x_train[source[0]]))
        axes[i+2].set_title(f"A - Source {source[0]} ({blendedSourceImageActivation[1][i]:.4f}): {image[1]}")
        axes[i+2].imshow(Image.fromarray(image[0].cpu().numpy()*255))
    plt.show()

    # Display closestMostUsedSourceImagesSum
    fig, axes = plt.subplots(1, len(closestMostUsedSourceImagesSum)+2, figsize=(35, 35))
    axes[0].set_title(f"LINEAR - Original: {originalImage[1]}")
    axes[0].imshow(Image.fromarray(originalImage[0].cpu().numpy()*255))
    axes[1].set_title(f"S - Closest Sources/Neuron (Most Used)")
    axes[1].imshow(blendedSourceImageSum[0])
    for i, source in enumerate(closestMostUsedSourceImagesSum):
        image = createImageWithPrediction(x_train[source[0]], y_train[source[0]], predict(x_train[source[0]]))
        axes[i+2].set_title(f"S - Source {source[0]} ({blendedSourceImageSum[1][i]:.4f}x): {image[1]}")
        axes[i+2].imshow(Image.fromarray(image[0].cpu().numpy()*255))
    plt.show()

"""# Evaluation: Closest Sources"""

from dataclasses import dataclass
@dataclass(order=True)
class WeightedSource:
  source: int
  difference: float

def identifyClosestSources(dictionary, outputs, mode = ""):
    if(mode == "Sum"):
        layerNumbersToCheck = [idx*2 for idx, (name, layer, activation) in enumerate(layers)]
    elif(mode == "Activation"):
        layerNumbersToCheck = [(idx*2)+1 for idx, (name, layer, activation) in enumerate(layers) if getActivation(hidden_sizes, idx) != False]

    layersToCheck = dictionary[layerNumbersToCheck]
    outputsToCheck = outputs[layerNumbersToCheck]
    identifiedClosestSources = np.empty((len(layersToCheck), np.max(layerSizes), closestSources), dtype=tuple)

    #print(len(layersToCheck), len(outputsToCheck))
    for currentLayer, layer in enumerate(layersToCheck):
        for currentNeuron, neuron in enumerate(layer):
            if(currentNeuron < layers[currentLayer][1].out_features):
                #print(currentLayer, currentNeuron, len(neuron), outputsToCheck[currentLayer][currentNeuron], outputsToCheck.shape)
                differencesBetweenSources = np.abs(neuron - np.full(len(neuron), outputsToCheck[currentLayer][currentNeuron]))
                sortedSourceIndices = np.argsort(differencesBetweenSources) # for highest difference uncomment this: [::-1]
                closestSourceIndices = sortedSourceIndices[:closestSources]
                tuples = tuple((closestSourceIndices[i], neuron[closestSourceIndices[i]], abs(neuron[closestSourceIndices[i]]-outputsToCheck[currentLayer][currentNeuron])) for i in range(closestSources))
                identifiedClosestSources[currentLayer][currentNeuron] = tuples

    return identifiedClosestSources, outputsToCheck, layerNumbersToCheck

def getMostUsed(sources):
    mostUsed = []
    sourceCounter = 0
    for currentLayer, layer in enumerate(sources):
        for currentNeuron, neuron in enumerate(layer):
            if(currentNeuron < layers[currentLayer][1].out_features):
                for sourceNumber, value, difference in neuron:
                    mostUsed.append(sourceNumber)
                    sourceCounter += 1
    return sourceCounter, mostUsed

def getMostUsedSources(sources, weightedMode=""):
    weightedSources = []

    sourceCounter, mostUsed = getMostUsed(sources)
    counter = Counter(mostUsed)

    print(sourceCounter, counter.most_common())
    return counter.most_common()[:closestSources]

def getMostUsedPerLayer(sources):
    mostUsed = []
    sourceCounter = 0
    for src in sources:
        mostUsed.append(src.source)
        sourceCounter += 1
    return sourceCounter, mostUsed

def getClosestSourcesPerNeuronAndLayer(sources, layersToCheck, mode=""):
    for cLayer, layer in enumerate(sources):
        weightedSourcesPerLayer = []
        totalDifferencePerLayer = 0
        imagesPerLayer = []

        for cNeuron, neuron in enumerate(layer):
            if(cNeuron < layers[cLayer][1].out_features):
                weightedSourcesPerNeuron = []
                totalDifferencePerNeuron = 0
                for sourceNumber, value, difference in neuron:
                    baseWeightedSource = {'source': sourceNumber, 'difference': difference}
                    totalDifferencePerNeuron += difference
                    totalDifferencePerLayer += difference
                    weightedSourcesPerNeuron.append(WeightedSource(**baseWeightedSource))
                    weightedSourcesPerLayer.append(WeightedSource(**baseWeightedSource))
                if not(visualizationChoice.value == "Custom" and ((cNeuron < int(visualizeCustom[cLayer][0][0])) or (cNeuron > int(visualizeCustom[cLayer][0][1])))):
                    imagesPerLayer.append([blendIndividualImagesTogether(weightedSourcesPerNeuron), [f"Source: {source.source}, Difference: {source.difference:.10f}<br>" for source in weightedSourcesPerNeuron][:showClosestMostUsedSources], f"{mode} - Layer: {int(layersToCheck[cLayer]/2)}, Neuron: {cNeuron}"])

        if not(visualizationChoice.value == "Per Layer Only"):
            if not(mode == "Activation" and visualizationChoice.value == "Custom" and visualizeCustom[cLayer][1] == False):
                showIndividualImagesPlotly(imagesPerLayer, int(layersToCheck[cLayer]/2), mode)

        if not(visualizationChoice.value == "Per Neuron Only"):
            if not(mode == "Activation" and visualizationChoice.value == "Custom" and visualizeCustom[cLayer][1] == False):
                weightedSourcesPerLayer = sorted(weightedSourcesPerLayer, key=lambda x: x.difference)
                sourceCounter, mostUsed = getMostUsedPerLayer(weightedSourcesPerLayer)
                counter = Counter(mostUsed)
                image = blendIndividualImagesTogether(counter.most_common()[:closestSources], True)

                plt.figure(figsize=(28,28))
                plt.imshow(image)
                plt.title(f"{mode} - Layer:  {int(layersToCheck[cLayer]/2)}, {closestSources} most used Sources")
                plt.show()

"""# Evaluation: Visual Blending"""

def blendIndividualImagesTogether(mostUsedSources, layer=False):
    image = Image.fromarray(np.zeros(shape=[28,28], dtype=np.uint8)).convert("RGBA")

    total = 0
    for source in mostUsedSources:
        if(layer):
            total += source[1]
        else:
            total += source.difference

    for wSource in mostUsedSources:
        #TODO: NORMALIZATION!!!
        if(total > 0):
            if(closestSources < 2):
                if(layer):
                    image = Image.blend(image, Image.fromarray(x_train[wSource[0]].numpy()*255).convert("RGBA"), 1)
                else:
                    image = Image.blend(image, Image.fromarray(x_train[wSource.source].numpy()*255).convert("RGBA"), 1)
            else:
                if(layer):
                    image = Image.blend(image, Image.fromarray(x_train[wSource[0]].numpy()*255).convert("RGBA"), wSource[1] / total)
                else:
                    #print(f"Diff: {wSource.difference}, Total: {total}, Calculation: {(1 - (wSource.difference / total)) / closestSources}")
                    image = Image.blend(image, Image.fromarray(x_train[wSource.source].numpy()*255).convert("RGBA"), (1 - (wSource.difference / total)) / closestSources)

    return image

"""# Evaluation: Prediction"""

def predict(sample):
    with torch.no_grad():
        sample = sample.to(device)
        model.eval()
        output = model(torch.flatten(sample))
    normalizedPredictions = normalizePredictions(output.cpu().numpy())
    if(datasetChoice.value == "HSV-RGB"):
        return output, 1.0
    return np.argmax(normalizedPredictions), normalizedPredictions[np.argmax(normalizedPredictions)]

def createImageWithPrediction(sample, true, prediction):
    sample = sample.to(device)
    true = true.to(device)
    prediction, probability = predict(sample)
    true_class = int(torch.argmax(true.cpu()))  # Move `true` tensor to CPU and then get the index of the maximum value
    return [sample, f"pred: {prediction}, prob: {probability:.2f}, true: {true_class}"]

def normalizePredictions(array):
    min = np.min(array)
    max = np.max(array)
    return (array - min) / (max - min)

"""# Evaluation: Code"""

#Make sure to set new dictionarys for the hooks to fill - they are global!
dictionaryForSourceLayerNeuron = np.zeros((eval_samples, len(layers)*2, np.max(layerSizes)))
dictionaryForLayerNeuronSource = np.zeros((len(layers)*2, np.max(layerSizes), eval_samples))

with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    attachHooks(eval_dataloader)

for pos, (sample, true) in enumerate(eval_dataloader):
    sample = sample.float()
    prediction = predict(sample)

    if(visualizationChoice.value == "Weighted" and datasetChoice.value == "MNIST"):
        sourcesSum, outputsSum, layerNumbersToCheck = identifyClosestSources(activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Sum")

        mostUsedSourcesWithSum = getMostUsedSources(sourcesSum, "Sum")
        blendedSourceImageSum = blendImagesTogether(mostUsedSourcesWithSum[:20], "Not Weighted")

        sourcesActivation, outputsActivation, layerNumbersToCheck = identifyClosestSources(activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Activation")
        mostUsedSourcesWithActivation = getMostUsedSources(sourcesActivation, "Activation")
        blendedSourceImageActivation = blendImagesTogether(mostUsedSourcesWithActivation[:20], "Not Weighted")

        showImagesUnweighted(createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedSourceImageActivation, blendedSourceImageSum, mostUsedSourcesWithActivation[:showClosestMostUsedSources], mostUsedSourcesWithSum[:showClosestMostUsedSources])
    elif(datasetChoice.value == "MNIST"):
        sourcesSum, outputsSum, layerNumbersToCheck = identifyClosestSources(activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Sum")
        mostUsedSourcesWithSum = getClosestSourcesPerNeuronAndLayer(sourcesSum, layerNumbersToCheck, "Sum")

        sourcesActivation, outputsActivation, layerNumbersToCheck = identifyClosestSources(activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Activation")
        mostUsedSourcesWithActivation = getClosestSourcesPerNeuronAndLayer(sourcesActivation, layerNumbersToCheck, "Activation")

print(f"Time passed since start: {time_since_start(startTime)}")