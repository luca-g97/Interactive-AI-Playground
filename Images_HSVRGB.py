# import time

# from keras.datasets import mnist
# from keras.utils import to_categorical

import colorsys
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import numpy as np
# import random

import torch
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torch import nn
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Subset

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(device)

import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = 'colab'

# def time_since_start(start_time):
#     current_time = time.time()
#     elapsed_time = current_time - start_time

#     hours = int(elapsed_time // 3600)
#     minutes = int((elapsed_time % 3600) // 60)
#     seconds = int(elapsed_time % 60)
#     milliseconds = int((elapsed_time % 1) * 1000)

#     return f"{hours}h {minutes}m {seconds}s {milliseconds}ms"

# """#Initialization"""

def generate_equidistant_color_samples(n_samples):
    # Calculate the cube root - approximate steps per dimension
    steps_per_dimension = round(n_samples ** (1/3))

    # Calculate the step size for each dimension, scaling down to 0-1 range
    step_size = 1 / (steps_per_dimension - 1)

    # Generate the RGB values scaled to 0-1
    r_values = np.arange(0, 1.001, step_size)
    g_values = np.arange(0, 1.001, step_size)
    b_values = np.arange(0, 1.001, step_size)

    # Create a grid of RGB values
    rgb_array = np.array(np.meshgrid(r_values, g_values, b_values)).T.reshape(-1,3)
    array = np.zeros([len(rgb_array), 2, 3])
    colorArray = np.zeros([len(rgb_array), 1, 3])

    for x in range(len(rgb_array)): # merge the rgb and hsv values
        hsv = colorsys.rgb_to_hsv(rgb_array[x][0],rgb_array[x][1],rgb_array[x][2])
        hsv = [float(color) for color in hsv]
        rgb = [float(color) for color in rgb_array[x]]
        array[x] = (hsv, rgb)

    

    return [(torch.from_numpy(x[0]), torch.from_numpy(x[1])) for x in array]

def generate_random_color_samples(n_samples):
    data = np.empty([n_samples, 2, 3])

    for x in range(n_samples):
        hsv = np.random.random(3)
        hsv = [float(color) for color in hsv]
        rgb = colorsys.hsv_to_rgb(hsv[0],hsv[1],hsv[2])
        rgb = [float(color) for color in rgb]
        data[x] = (hsv, rgb)

    return [(torch.from_numpy(x[0]), torch.from_numpy(x[1])) for x in data]

def createTrainAndTestSet(trainSetLength, testSetLength):
    trainSetHSVRGB = generate_equidistant_color_samples(trainSetLength)
    testSetHSVRGB = generate_random_color_samples(testSetLength)

    fig = showSamplesCube(trainSetHSVRGB, testSetHSVRGB)

    return torch.from_numpy(trainSetHSVRGB), torch.from_numpy(testSetHSVRGB), fig

def draw_RGB_3D(array, traceName):
    # Extracting HSV and RGB components from the array
    hsv_values = [hsv for hsv, rgb in array]
    rgb_values = [rgb for hsv, rgb in array]

    # Create a scatter plot
    scatter = go.Scatter3d(
        x=[rgb[0] for rgb in rgb_values],
        y=[rgb[1] for rgb in rgb_values],
        z=[rgb[2] for rgb in rgb_values],
        mode='markers',
        marker=dict(
            size=2,
            color=[f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})' for rgb in rgb_values],  # Convert RGB values to string
            opacity=0.8
        ),
        name=traceName
    )

    # Create the figure layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        title=f'Train Samples: {len(array)}'
    )

    # Create the figure and plot
    fig = go.Figure(data=[scatter], layout=layout)
    return fig

def showSamplesCube(trainSetHSVRGB, testSetHSVRGB):
    train = [(hsv, rgb) for hsv, rgb in trainSetHSVRGB]
    test = [(hsv, rgb) for hsv, rgb in testSetHSVRGB]
    array = (train, test)
    names = ("Training", "Test")

    # Create multiple plots side by side
    plots = []
    for i in range(2):  # Assuming you want to display 3 plots side by side
        test_subset = array[i]  # Adjust the range as needed
        plots.append(draw_RGB_3D(test_subset, names[i]))

    # Display the plots side by side
    fig = go.Figure()
    for i, plot in enumerate(plots):
        fig.add_trace(plot.data[0])

    fig.update_layout(
        title='Train and Testsamples for HSV-RGB',
        width=1000,  # Adjust the total width as needed
        height=400,  # Adjust the height as needed
        grid=dict(rows=1, columns=len(plots), pattern='independent'),
    )

    # Save the plot as an HTML file
    #pio.write_html(fig, file='multiple_rgb_plots.html', auto_open=True)

    return fig

# """#Data Initialization"""

# class CustomMNISTData(Dataset):
#     def __init__(self, mode="", transform = None):
#         if mode=="Train":
#             self.input = x_train
#             self.output = y_train
#         elif mode=="Test":
#             self.input = x_test
#             self.output = y_test
#         elif mode=="Evaluate":
#             self.input = x_eval
#             self.output = y_eval

#         self.transform = transform

#     def __len__(self):
#         return len(self.input)

#     def __getitem__(self, idx):
#         image = self.input[idx]
#         output = self.output[idx]

#         return torch.flatten(image), output

# def getLayer(hidden_layers, layer, input_size=10, output_size=10):
#     if(hidden_layers[layer][0] == "Linear"):
#         return nn.Linear(input_size, output_size)
#     elif(hidden_layers[layer][0] == "Conv2d"):
#         return nn.Conv2d(input_size, output_size)
#     return False

# def getActivation(hidden_layers, layer):
#     if(hidden_layers[layer][2] == "ReLU"):
#         return nn.ReLU()
#     elif(hidden_layers[layer][2] == "Sigmoid"):
#         return nn.Sigmoid()
#     elif(hidden_layers[layer][2] == "Tanh"):
#         return nn.Tanh()
#     return False

# def checkIfActivationLayerExists(hidden_layers, layer):
#     if hidden_layers[layer][2] != "None":
#         return True
#     return False

# layers = []
# currentLayer = 0
# relevantLayerIndices = []
# def createLayers(layerName, layerType, activationLayerType):
#     global currentLayer

#     layers.append((layerName, layerType, activationLayerType))
#     relevantLayerIndices.append(currentLayer*2)
#     if(activationLayerType != "None"):
#         relevantLayerIndices.append((currentLayer*2)+1)
#     currentLayer += 1
#     return layerName, layerType

# class CustomizableRENN(nn.Module):
#     def __init__(self, input_size, hidden_layers, output_size):
#         super(CustomizableRENN, self).__init__()
#         #Add input and output layer to the hidden_layers
#         self.num_layers = (len(hidden_sizes))
#         self.hidden_layers = hidden_layers

#         for layer in range(self.num_layers):
#             if layer == 0:

#                 setattr(self, *createLayers(f'fc{layer}', getLayer(self.hidden_layers, layer, input_size, hidden_layers[layer][1]), self.hidden_layers[layer][2]))
#             elif layer == (self.num_layers - 1):
#                 setattr(self, *createLayers(f'fc{layer}', getLayer(self.hidden_layers, layer, hidden_layers[layer-1][1], output_size), self.hidden_layers[layer][2]))
#             else:
#                 #print(layer, layer-1, self.num_layers, hidden_sizes[layer-1], hidden_sizes[layer])
#                 setattr(self, *createLayers(f'fc{layer}', getLayer(self.hidden_layers, layer, hidden_layers[layer-1][1], hidden_layers[layer][1]), self.hidden_layers[layer][2]))

#             if checkIfActivationLayerExists(self.hidden_layers, layer):
#                 setattr(self, f'activation{layer}', getActivation(self.hidden_layers, layer))

#     def forward(self, x):
#         for layer in range(self.num_layers):
#             x = getattr(self, f'fc{layer}')(x)

#             #There can not be an activation layer at the input and output!
#             if (checkIfActivationLayerExists(self.hidden_layers, layer)):
#                 x = getattr(self, f'activation{layer}')(x)
#         return x

# input_size = len(torch.flatten(torch.tensor(train_dataloader.dataset[0][0])))
# output_size = len(torch.flatten(torch.tensor(train_dataloader.dataset[0][1])))

# model = CustomizableRENN(input_size, hidden_sizes, output_size)
# model.to(device)
# layers = np.array(layers)

# if(loss_function == "MSE"):
#     criterion_class = nn.MSELoss()  # For regression
# elif(loss_function == "Cross-Entropy"):
#     criterion_class = nn.CrossEntropyLoss()  # For multi-class classification

# if(optimizer == "Adam"):
#     chosen_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# elif(optimizer == "SGD"):
#     chosen_optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# # Training Loop
# def train_model(model, criterion_class,  optimizer, train_dataloader, test_dataloader, epochs=10):
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0.0
#         #print(len(train_dataloader.dataset[0][0]))
#         for images, classification in train_dataloader:
#             images = images.float()
#             images = images.to(device)
#             classification = classification.float()
#             classification = classification.to(device)
#             optimizer.zero_grad()
#             output = model(images)
#             loss = criterion_class(output, classification)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         # Validation loop
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for images, classification in test_dataloader:
#                 images = images.float()
#                 images = images.to(device)
#                 classification = classification.float()
#                 classification = classification.to(device)
#                 output = model(images)
#                 loss = criterion_class(output, classification)
#                 val_loss += loss.item()

#         # Print statistics
#         print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_dataloader)}, Validation Loss: {val_loss/len(test_dataloader)}')

# # Train the model
# trainTime = time.time()
# train_model(model, criterion_class, chosen_optimizer, train_dataloader, test_dataloader, epochs=epochs)
# print(f"Time passed since start of training: {time_since_start(trainTime)}")

# layerSizes = [size[1] for size in hidden_sizes[:]]
# activationsBySources = np.full((train_samples, len(layers)*2, np.max(layerSizes)))
# activationsByLayers = np.full((len(layers)*2, np.max(layerSizes), train_samples), )
# print(activationsBySources.shape)
# print(activationsByLayers.shape)

# #Variables for usage within the hook
# layer = 0
# source = 0
# dictionaryForSourceLayerNeuron = activationsBySources
# dictionaryForLayerNeuronSource = activationsByLayers
# train_dataloader = DataLoader(trainSubset, batch_size=1, shuffle=False)
# print(layers)

# # Forward hook
# def forward_hook(module, input, output):
#     global layer
#     global source
#     global dictionaryForSourceLayerNeuron
#     global dictionaryForLayerNeuronSource
#     global hidden_sizes

#     activation_type = type(getActivation(hidden_sizes, int(layer/2)))
#     layer_type = type(getLayer(hidden_sizes, int(layer/2)))
#     relevantOutput = output[0].cpu().numpy()

#     if (type(module) == activation_type or type(module) == layer_type):
#         #Use for array structure like: [source, layer, neuron]
#         dictionaryForSourceLayerNeuron[source][layer,:layers[int(layer/2)][1].out_features] = relevantOutput

#         #Use for array structure like: [layer, neuron, source]
#         for neuronNumber, neuron in enumerate(relevantOutput):
#             if neuronNumber < layers[int(layer/2)][1].out_features:
#                 dictionaryForLayerNeuronSource[layer][neuronNumber][source] = neuron
#             else:
#                 break

#     if(layer % 2 == 0):
#         if(checkIfActivationLayerExists(hidden_sizes, int(layer/2))):
#             layer += 1
#         elif(layer == (len(layers)*2)-2):
#             layer = 0
#         else:
#             layer += 2
#     else:
#         if(layer == (len(layers)*2)-1):
#             layer = 0
#         else:
#             layer += 1
# #-------------------------------------------------------------------------------

# def attachHooks(hookLoader):
#     global source

#     hooks = []  # Store the handles for each hook
#     outputs = np.array([])

#     for name, module in model.named_modules():
#       if not isinstance(module, CustomizableRENN):
#           hook = module.register_forward_hook(forward_hook)
#           hooks.append(hook)

#     with torch.no_grad():
#       # Forward Pass
#       convertTime = time.time()
#       for source, (inputs, labels) in enumerate(hookLoader):
#         # Uncomment for array structure like: [source, layer, neuron]
#         inputs = inputs.float()
#         inputs = inputs.to(device)
#         _ = model(inputs)

#       print(f"Time passed since conversion: {time_since_start(convertTime)}")

#     # Remove hooks after use
#     for hook in hooks:
#         hook.remove()

# attachHooks(train_dataloader)
# activationsBySources = dictionaryForSourceLayerNeuron
# activationsByLayers = dictionaryForLayerNeuronSource

# activationsByLayers[0][0]

# import matplotlib.pyplot as plt
# from collections import Counter
# from PIL import Image

# def showImagesUnweighted(originalImage, blendedSourceImageActivation, blendedSourceImageSum, closestMostUsedSourceImagesActivation, closestMostUsedSourceImagesSum):
#     fig, axes = plt.subplots(1, 5, figsize=(35, 35))
#     plt.subplots_adjust(hspace=0.5)

#     # Display original image
#     axes[0].set_title(f"BLENDED - Original: {originalImage[1]}")
#     axes[0].imshow(Image.fromarray(originalImage[0].cpu().numpy()*255))

#     # Display blendedSourceImageActivation
#     axes[1].set_title(f"A=Activation - Closest Sources/Neuron (Most Used)")
#     axes[1].imshow(blendedSourceImageActivation[0])

#     # Display blendedSourceImageSum
#     axes[2].set_title(f"S=Sum - Closest Sources/Neuron (Most Used)")
#     axes[2].imshow(blendedSourceImageSum[0])

#     # Display weightedSourceImageActivation
#     axes[3].set_title(f"WA=WeightedActivation - Closest Sources/Neuron (Weighted)")
#     axes[3].imshow(Image.fromarray(np.zeros(shape=[28,28], dtype=np.uint8)).convert("RGBA"))

#     # Display weightedSourceImageSum
#     axes[4].set_title(f"WS=WeigthedSum - Closest Sources/Neuron (Weighted)")
#     axes[4].imshow(Image.fromarray(np.zeros(shape=[28,28], dtype=np.uint8)).convert("RGBA"))

#     plt.show()

#     # Display closestMostUsedSourceImagesActivation
#     fig, axes = plt.subplots(1, len(closestMostUsedSourceImagesActivation)+2, figsize=(35, 35))
#     axes[0].set_title(f"NON-LINEAR - Original: {originalImage[1]}")
#     axes[0].imshow(Image.fromarray(originalImage[0].cpu().numpy()*255))
#     axes[1].set_title(f"A - Closest Sources/Neuron (Most Used)")
#     axes[1].imshow(blendedSourceImageActivation[0])
#     for i, source in enumerate(closestMostUsedSourceImagesActivation):
#         image = createImageWithPrediction(x_train[source[0]], y_train[source[0]], predict(x_train[source[0]]))
#         axes[i+2].set_title(f"A - Source {source[0]} ({blendedSourceImageActivation[1][i]:.4f}): {image[1]}")
#         axes[i+2].imshow(Image.fromarray(image[0].cpu().numpy()*255))
#     plt.show()

#     # Display closestMostUsedSourceImagesSum
#     fig, axes = plt.subplots(1, len(closestMostUsedSourceImagesSum)+2, figsize=(35, 35))
#     axes[0].set_title(f"LINEAR - Original: {originalImage[1]}")
#     axes[0].imshow(Image.fromarray(originalImage[0].cpu().numpy()*255))
#     axes[1].set_title(f"S - Closest Sources/Neuron (Most Used)")
#     axes[1].imshow(blendedSourceImageSum[0])
#     for i, source in enumerate(closestMostUsedSourceImagesSum):
#         image = createImageWithPrediction(x_train[source[0]], y_train[source[0]], predict(x_train[source[0]]))
#         axes[i+2].set_title(f"S - Source {source[0]} ({blendedSourceImageSum[1][i]:.4f}x): {image[1]}")
#         axes[i+2].imshow(Image.fromarray(image[0].cpu().numpy()*255))
#     plt.show()

# def showIndividualImagesHSVRGB(images):
#     # Define the number of rows and columns for subplots
#     num_images = len(images)
#     num_cols =  5 # Number of columns
#     if(len(images) == 10):
#         num_cols = 5
#     num_rows = num_images // num_cols # Number of rows
#     if(num_images%num_cols != 0):
#         num_rows = num_images // num_cols + 1 # Number of rows

#     # Create a figure and subplots
#     fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
#     plt.subplots_adjust(hspace=0.5)

#     # Flatten the axes if it's a single row or column
#     if num_rows == 1:
#         axs = axs.reshape(1, -1)
#     if num_cols == 1:
#         axs = axs.reshape(-1, 1)

#     # Plot each image
#     for i in range(num_images):
#         row_index = i // num_cols
#         col_index = i % num_cols
#         axs[row_index, col_index].imshow(np.array(Image.open(io.BytesIO(images[i][0]))))
#         axs[row_index, col_index].set_title(images[i][1], fontweight='bold')
#         axs[row_index, col_index].axis('off')

#     #Fill up with empty images if necessary
#     for i in range(num_images % num_cols):
#         image = Image.fromarray(np.ones(shape=[28,28], dtype=np.uint8)).convert("RGBA")
#         row_index = (num_images-1 + i) // num_cols
#         col_index = (num_images-1 + i) % num_cols
#         axs[row_index, col_index].imshow(image)
#         axs[row_index, col_index].axis('off')

#     # Adjust layout and show the plot
#     plt.tight_layout()
#     plt.show()

# import plotly.graph_objects as go
# import plotly.subplots as sp

# def showIndividualImagesPlotly(images, layer, mode):
#     num_images = len(images)
#     num_cols = 5  # Number of columns
#     if num_images == 10:
#         num_cols = 5
#     num_rows = num_images // num_cols if (num_images // num_cols >= 1) else 1  # Number of rows
#     if num_images % num_cols != 0:
#         num_rows = num_images // num_cols + 1  # Number of rows

#     fig = sp.make_subplots(rows=num_rows, cols=num_cols)

#     for i, (image, sources, title) in enumerate(images):
#         row_index = i // num_cols
#         col_index = i % num_cols
#         fig.add_trace(go.Image(name=f'<b>{title}</b><br>Closest {showClosestMostUsedSources} Sources compared by {mode}:<br>{sources}', z=image), row=row_index + 1, col=col_index + 1)

#     fig.update_layout(
#         title=f'Blended closest {closestSources} sources for each neuron in layer {layer} (compared to their {mode} output)',
#         grid={'rows': num_rows, 'columns': num_cols},
#         height=225 * num_rows,  # Adjust the height of the plot
#         width=225 * num_cols,
#         hoverlabel=dict(namelength=-1)
#     )

#     #fig.show()
#     display(fig)

# import io

# vectorsToShow = []

# def createComparison(hsv_sample, rgb_predicted, blendedHSV, blendedRGB, weighting):

#     fig, axs = plt.subplots(5, 1, figsize=(6, 6))
#     rgb_predicted = rgb_predicted.detach().numpy()[0]

#     original_rgb = colorsys.hsv_to_rgb(hsv_sample[0], hsv_sample[1], hsv_sample[2])

#     # Original color patch (HSV to RGB)
#     axs[0].add_patch(patches.Rectangle((0, 0), 1, 1, color=np.array(hsv_sample)))
#     axs[0].axis('off')
#     axs[0].set_title(f'HSV - Original: {tuple([int(x * 255) for x in hsv_sample])}')

#     blended_hsv = tuple(float("{:.2f}".format(x * 255)) for x in blendedHSV[0])
#     blendedHSV_difference = tuple(float("{:.2f}".format((x*255) - y)) for x, y in zip(hsv_sample, blended_hsv))
#     axs[1].add_patch(patches.Rectangle((0, 0), 1, 1, color=np.array(blendedHSV[0])))
#     axs[1].axis('off')
#     axs[1].set_title(f"HSV-Blended : {blended_hsv}\nDifference: {blendedHSV_difference}")

#     # Original color patch (HSV to RGB)
#     vectorsToShow.append([tuple([int(x * 255) for x in original_rgb]), 1, [255/255, 165/255, 0/255], "RGB-Reference"])
#     axs[2].add_patch(patches.Rectangle((0, 0), 1, 1, color=np.array(original_rgb)))
#     axs[2].axis('off')
#     axs[2].set_title(f'RGB - Original: {tuple([int(x * 255) for x in original_rgb])}')

#     # Predicted color patch
#     vectorsToShow.append([tuple([int(x * 255) for x in rgb_predicted]), 1, [0/255, 128/255, 128/255], "RGB-Predicted"])
#     difference = tuple(float("{:.2f}".format((x - y) * 255)) for x, y in zip(original_rgb, rgb_predicted))
#     axs[3].add_patch(patches.Rectangle((0, 0), 1, 1, color=np.array(rgb_predicted)))
#     axs[3].axis('off')
#     axs[3].set_title(f'RGB - Predicted: {tuple([int(x * 255) for x in rgb_predicted])}\nDifference: {difference}')

#     for source, weight in weighting:
#         vectorsToShow.append([tuple([float("{:.2f}".format(x * 255)) for x in trainDataSet[source][1]]), weight, trainDataSet[source][1], source])


#     blended_rgb = tuple(float("{:.2f}".format(x * 255)) for x in blendedRGB[0])
#     blendedRGBOriginal_difference = tuple(float("{:.2f}".format((x*255) - y)) for x, y in zip(original_rgb, blended_rgb))
#     blendedRGBPredicted_difference = tuple(float("{:.2f}".format((x*255) - y)) for x, y in zip(rgb_predicted, blended_rgb))
#     vectorsToShow.append([blended_rgb, 1, [0,0,0], f"RGB-Weighted"])
#     axs[4].add_patch(patches.Rectangle((0, 0), 1, 1, color=np.array(blendedRGB)))
#     axs[4].axis('off')
#     axs[4].set_title(f"RGB - Blended: {blended_rgb}\nOriginal->Blended: {blendedRGBOriginal_difference}\nPredicted->Blended: {blendedRGBPredicted_difference}")
#     plt.tight_layout()

#     # Save the plot as an image
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     plt.close(fig)
#     return buffer.getvalue()

# from dataclasses import dataclass
# @dataclass(order=True)
# class WeightedSource:
#   source: int
#   difference: float

# def getMostUsed(sources):
#     mostUsed = []
#     sourceCounter = 0
#     for currentLayer, layer in enumerate(sources):
#         for currentNeuron, neuron in enumerate(layer):
#             if(currentNeuron < layers[currentLayer][1].out_features):
#                 for sourceNumber, value, difference in neuron:
#                     mostUsed.append(sourceNumber)
#                     sourceCounter += 1
#     return sourceCounter, mostUsed

# def getMostUsedSources(sources, weightedMode=""):
#     weightedSources = []

#     sourceCounter, mostUsed = getMostUsed(sources)
#     counter = Counter(mostUsed)

#     print(sourceCounter, counter.most_common())
#     return counter.most_common()[:closestSources]

# def getMostUsedPerLayer(sources):
#     mostUsed = []
#     sourceCounter = 0
#     for src in sources:
#         mostUsed.append(src.source)
#         sourceCounter += 1
#     return sourceCounter, mostUsed

# def getClosestSourcesPerNeuronAndLayer(sources, layersToCheck, mode=""):
#     for cLayer, layer in enumerate(sources):
#         weightedSourcesPerLayer = []
#         totalDifferencePerLayer = 0
#         imagesPerLayer = []

#         for cNeuron, neuron in enumerate(layer):
#             if(cNeuron < layers[cLayer][1].out_features):
#                 weightedSourcesPerNeuron = []
#                 totalDifferencePerNeuron = 0
#                 for sourceNumber, value, difference in neuron:
#                     baseWeightedSource = {'source': sourceNumber, 'difference': difference}
#                     totalDifferencePerNeuron += difference
#                     totalDifferencePerLayer += difference
#                     weightedSourcesPerNeuron.append(WeightedSource(**baseWeightedSource))
#                     weightedSourcesPerLayer.append(WeightedSource(**baseWeightedSource))
#                 if not(visualizationChoice.value == "Custom" and ((cNeuron < int(visualizeCustom[cLayer][0][0])) or (cNeuron > int(visualizeCustom[cLayer][0][1])))):
#                     imagesPerLayer.append([blendIndividualImagesTogether(weightedSourcesPerNeuron), [f"Source: {source.source}, Difference: {source.difference:.10f}<br>" for source in weightedSourcesPerNeuron][:showClosestMostUsedSources], f"{mode} - Layer: {int(layersToCheck[cLayer]/2)}, Neuron: {cNeuron}"])

#         if not(visualizationChoice.value == "Per Layer Only"):
#             if not(mode == "Activation" and visualizationChoice.value == "Custom" and visualizeCustom[cLayer][1] == False):
#                 showIndividualImagesPlotly(imagesPerLayer, int(layersToCheck[cLayer]/2), mode)

#         if not(visualizationChoice.value == "Per Neuron Only"):
#             if not(mode == "Activation" and visualizationChoice.value == "Custom" and visualizeCustom[cLayer][1] == False):
#                 weightedSourcesPerLayer = sorted(weightedSourcesPerLayer, key=lambda x: x.difference)
#                 sourceCounter, mostUsed = getMostUsedPerLayer(weightedSourcesPerLayer)
#                 counter = Counter(mostUsed)
#                 image = blendIndividualImagesTogether(counter.most_common()[:closestSources], True)

#                 plt.figure(figsize=(28,28))
#                 plt.imshow(image)
#                 plt.title(f"{mode} - Layer:  {int(layersToCheck[cLayer]/2)}, {closestSources} most used Sources")
#                 plt.show()

# def blendIndividualImagesTogetherHSVRGB(mostUsedSources, layer=False):
#     hsv = np.zeros(shape=[1, 3], dtype=float)
#     rgb = np.zeros(shape=[1, 3], dtype=float)
#     weighting = []

#     total = 0
#     for source in mostUsedSources:
#         if(layer):
#             total += source[1]
#         else:
#             total += source.difference

#     for wSource in mostUsedSources:
#         #TODO: NORMALIZATION!!!
#         if(total > 0):
#             if(closestSources < 2):
#                 if(layer):
#                     hsv = (trainDataSet[wSource[0]][0])
#                     rgb = (trainDataSet[wSource[0]][1])
#                     weighting = [[wSource[0], 1]]
#                 else:
#                     hsv = (trainDataSet[wSource[0]][0])
#                     rgb = (trainDataSet[wSource[0]][1])
#                     weighting = [[wSource.source, 1]]
#             else:
#                 if(layer):
#                     hsv += np.concatenate((trainDataSet[wSource[0]][0],)) * (wSource[1] / total)
#                     rgb += np.concatenate((trainDataSet[wSource[0]][1],)) * (wSource[1] / total)
#                     weighting.append([wSource[0], wSource[1] / total])
#                 else:
#                     #print(f"Diff: {wSource.difference}, Total: {total}, Calculation: {(1 - (wSource.difference / total)) / closestSources}")
#                     hsv += np.concatenate((trainDataSet[wSource.source][0],)) * ((1 - (wSource.difference / total)) / closestSources)
#                     rgb += np.concatenate((trainDataSet[wSource.source][1],)) * ((1 - (wSource.difference / total)) / closestSources)
#                     weighting.append([wSource.source, (1 - (wSource.difference / total)) / closestSources])

#     return hsv, rgb, weighting

# def getClosestSourcesPerNeuronAndLayerHSVRGB(hsvSample, prediction, sources, mode=""):
#     for cLayer, layer in enumerate(sources):
#         weightedSourcesPerLayer = []
#         totalDifferencePerLayer = 0
#         imagesPerLayer = []

#         for cNeuron, neuron in enumerate(layer):
#             if(cNeuron < layers[cLayer][1].out_features):
#                 weightedSourcesPerNeuron = []
#                 totalDifferencePerNeuron = 0
#                 for sourceNumber, value, difference in neuron:
#                     baseWeightedSource = {'source': sourceNumber, 'difference': difference}
#                     totalDifferencePerNeuron += difference
#                     totalDifferencePerLayer += difference
#                     weightedSourcesPerNeuron.append(WeightedSource(**baseWeightedSource))
#                     weightedSourcesPerLayer.append(WeightedSource(**baseWeightedSource))
#                 if not(visualizationChoice.value == "Custom" and ((cNeuron < int(visualizeCustom[cLayer][0][0])) or (cNeuron > int(visualizeCustom[cLayer][0][1])))):
#                     hsv, rgb, weighting = blendIndividualImagesTogetherHSVRGB(weightedSourcesPerNeuron)
#                     neuronImage = createComparison(hsvSample[0], prediction[0], hsv, rgb, weighting)
#                     sortMode = mode
#                     if(mode == "Activation"):
#                         sortMode = "Act"
#                     imagesPerLayer.append([neuronImage, f"{sortMode} - Layer: {cLayer}, Neuron: {cNeuron}"])

#         if not(visualizationChoice.value == "Per Layer Only"):
#             if not(mode == "Activation" and visualizationChoice.value == "Custom" and visualizeCustom[cLayer][1] == False):
#                 showIndividualImagesHSVRGB(imagesPerLayer)

#         if not(visualizationChoice.value == "Per Neuron Only"):
#             if not(mode == "Activation" and visualizationChoice.value == "Custom" and visualizeCustom[cLayer][1] == False):
#                 weightedSourcesPerLayer = sorted(weightedSourcesPerLayer, key=lambda x: x.difference)
#                 sourceCounter, mostUsed = getMostUsedPerLayer(weightedSourcesPerLayer)
#                 counter = Counter(mostUsed)
#                 hsv, rgb, weighting = blendIndividualImagesTogetherHSVRGB(counter.most_common()[:closestSources], True)
#                 image = createComparison(hsvSample[0], prediction[0], hsv, rgb, weighting)

#                 plt.figure(figsize=(28,28))
#                 plt.imshow(np.array(Image.open(io.BytesIO(image))))
#                 plt.title(f"{mode} - Layer: {cLayer}, {closestSources} most used Sources")
#                 plt.show()

# def identifyClosestSources(dictionary, outputs, mode = ""):
#     if(mode == "Sum"):
#         layerNumbersToCheck = [idx*2 for idx, (name, layer, activation) in enumerate(layers)]
#     elif(mode == "Activation"):
#         layerNumbersToCheck = [(idx*2)+1 for idx, (name, layer, activation) in enumerate(layers) if getActivation(hidden_sizes, idx) != False]

#     layersToCheck = dictionary[layerNumbersToCheck]
#     outputsToCheck = outputs[layerNumbersToCheck]
#     identifiedClosestSources = np.empty((len(layersToCheck), np.max(layerSizes), closestSources), dtype=tuple)

#     #print(len(layersToCheck), len(outputsToCheck))
#     for currentLayer, layer in enumerate(layersToCheck):
#         for currentNeuron, neuron in enumerate(layer):
#             if(currentNeuron < layers[currentLayer][1].out_features):
#                 #print(currentLayer, currentNeuron, len(neuron), outputsToCheck[currentLayer][currentNeuron], outputsToCheck.shape)
#                 differencesBetweenSources = np.abs(neuron - np.full(len(neuron), outputsToCheck[currentLayer][currentNeuron]))
#                 sortedSourceIndices = np.argsort(differencesBetweenSources) # for highest difference uncomment this: [::-1]
#                 closestSourceIndices = sortedSourceIndices[:closestSources]
#                 tuples = tuple((closestSourceIndices[i], neuron[closestSourceIndices[i]], abs(neuron[closestSourceIndices[i]]-outputsToCheck[currentLayer][currentNeuron])) for i in range(closestSources))
#                 identifiedClosestSources[currentLayer][currentNeuron] = tuples

#     return identifiedClosestSources, outputsToCheck, layerNumbersToCheck

# def predict(sample):
#     with torch.no_grad():
#         sample = sample.to(device)
#         model.eval()
#         output = model(torch.flatten(sample))
#     normalizedPredictions = normalizePredictions(output.cpu().numpy())
#     if(datasetChoice.value == "HSV-RGB"):
#         return output, 1.0
#     return np.argmax(normalizedPredictions), normalizedPredictions[np.argmax(normalizedPredictions)]

# def createImageWithPrediction(sample, true, prediction):
#     sample = sample.to(device)
#     true = true.to(device)
#     prediction, probability = predict(sample)
#     true_class = int(torch.argmax(true.cpu()))  # Move `true` tensor to CPU and then get the index of the maximum value
#     return [sample, f"pred: {prediction}, prob: {probability:.2f}, true: {true_class}"]

# def normalizePredictions(array):
#     min = np.min(array)
#     max = np.max(array)
#     return (array - min) / (max - min)

# #Make sure to set new dictionarys for the hooks to fill - they are global!
# dictionaryForSourceLayerNeuron = np.zeros((eval_samples, len(layers)*2, np.max(layerSizes)))
# dictionaryForLayerNeuronSource = np.zeros((len(layers)*2, np.max(layerSizes), eval_samples))

# with torch.no_grad():
#     model.eval()  # Set the model to evaluation mode
#     attachHooks(eval_dataloader)

# for pos, (sample, true) in enumerate(eval_dataloader):
#     sample = sample.float()
#     prediction = predict(sample)

#     if(visualizationChoice.value == "Weighted" and datasetChoice.value == "MNIST"):
#         sourcesSum, outputsSum, layerNumbersToCheck = identifyClosestSources(activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Sum")

#         mostUsedSourcesWithSum = getMostUsedSources(sourcesSum, "Sum")
#         blendedSourceImageSum = blendImagesTogether(mostUsedSourcesWithSum[:20], "Not Weighted")

#         sourcesActivation, outputsActivation, layerNumbersToCheck = identifyClosestSources(activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Activation")
#         mostUsedSourcesWithActivation = getMostUsedSources(sourcesActivation, "Activation")
#         blendedSourceImageActivation = blendImagesTogether(mostUsedSourcesWithActivation[:20], "Not Weighted")

#         showImagesUnweighted(createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedSourceImageActivation, blendedSourceImageSum, mostUsedSourcesWithActivation[:showClosestMostUsedSources], mostUsedSourcesWithSum[:showClosestMostUsedSources])
#     else:
#         sourcesSum, outputsSum, layerNumbersToCheck = identifyClosestSources(activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Sum")
#         getClosestSourcesPerNeuronAndLayerHSVRGB(sample, createImageWithPrediction(sample, true, prediction), sourcesSum, "Sum")

#         sourcesActivation, outputsActivation, layerNumbersToCheck = identifyClosestSources(activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Activation")
#         getClosestSourcesPerNeuronAndLayerHSVRGB(sample, createImageWithPrediction(sample, true, prediction), sourcesActivation, "Activation")

# print(f"Time passed since start: {time_since_start(startTime)}")

# # @title Click `Show code` in the code cell. { display-mode: "form" }

# if(datasetChoice.value == "HSV-RGB"):
#     layerNumber = 1 # @param {type:"slider", min:0, max:2, step:1}
#     neuronNumber = 0 # @param {type:"slider", min:0, max:127, step:1}

#     import plotly.graph_objects as go
#     import numpy as np

#     neuronsInLayer = 128
#     if(layerNumber == 2 & neuronNumber > 9):
#         neuronNumber = 9

#     size = 5
#     xValues = []
#     yValues = []
#     zValues = []
#     weightValues = []
#     colourValues = []
#     textValues = []

#     pos = layerNumber * neuronsInLayer * (3+closestSources) + neuronNumber * (3+closestSources)
#     for data in vectorsToShow[pos:(pos+(3+closestSources))]:
#         xValues.append(data[0][0])
#         yValues.append(data[0][1])
#         zValues.append(data[0][2])
#         if(data[1] < 0.1):
#             weightValues.append(int((data[1]) * 200) + 5)
#         else:
#             weightValues.append(int((data[1]) * size + 5))
#         colourValues.append(data[2])
#         if ((data[3] != "RGB-Weighted") & (data[3] != "RGB-Predicted") & (data[3] != "RGB-Reference")):
#             textValues.append(f"Source {data[3]} (Weight: {data[1]*100:.2f}%)")
#         else:
#             textValues.append(data[3])

#     layout = go.Layout(
#         scene=dict(
#             camera=dict(
#                 eye=dict(x=1, y=1, z=1)
#             ),
#             aspectratio=dict(x=1, y=1, z=1)
#         )
#     )

#     fig = go.Figure(data=[go.Scatter3d(
#         x=xValues,
#         y=yValues,
#         z=zValues,
#         text=textValues,
#         mode='markers',
#         marker=dict(
#             size=weightValues,
#             color=colourValues,
#             opacity=0.8
#         )
#     )], layout=layout)

#     # tight layout
#     fig.update_layout(
#         scene = dict(
#             xaxis = dict(nticks=16, range=[0,256],),
#             yaxis = dict(nticks=16, range=[0,256],),
#             zaxis = dict(nticks=16, range=[0,256],),),
#         width=700,
#         margin=dict(r=20, l=10, b=10, t=40),
#         title=f"Layer: {layerNumber}, Neuron: {neuronNumber} ({closestSources} closest Sources)",
#         title_font_size=20)

#     display(fig)
