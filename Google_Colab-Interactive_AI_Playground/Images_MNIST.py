import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch.optim as optim
import numpy as np
import Customizable_RENN as RENN

mnist, to_categorical, nn, DataLoader, device = "", "", "", "", ""
train_dataloader, test_dataloader, eval_dataloader, trainDataSet, testDataSet, trainSubset, testSubset, x_train, y_train, x_test, y_test, x_eval, y_eval = "", "", "", "", "", "", "", "", "", "", "", "", ""
model, criterion_class, chosen_optimizer, layers = "", "", "", ""
train_samples, eval_samples, test_samples = 1, 1, 1
dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource = [], []

def initializePackages(mnistPackage, to_categoricalPackage, nnPackage, DataLoaderPackage, devicePackage):
    global mnist, to_categorical, nn, DataLoader, device
    mnist, to_categorical, nn, DataLoader, device = mnistPackage, to_categoricalPackage, nnPackage, DataLoaderPackage, devicePackage

def createTrainAndTestSet():
    global trainDataSet, testDataSet, x_train, y_train
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
    
    x_eval = x_test
    y_eval = y_test
    x_eval = torch.from_numpy(x_eval)
    y_eval = torch.from_numpy(y_eval)
    
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    
    trainDataSet = [(torch.flatten(x), torch.flatten(y)) for x, y in zip(x_train, y_train)]
    testDataSet = [(torch.flatten(x), torch.flatten(y)) for x, y in zip(x_test, y_test)]
    
    print(f"Created {len(trainDataSet)} Trainsamples & {len(testDataSet)} Testsamples")
    return trainDataSet, testDataSet

def initializeDatasets(train_samplesParameter, test_samplesParameter, eval_samplesParameter, batch_size_training, batch_size_test, seed=""):
    global train_samples, test_samples, eval_samples, np, torch
    global train_dataloader, test_dataloader, eval_dataloader, trainSubset, testSubset
    train_samples, test_samples, eval_samples = train_samplesParameter, test_samplesParameter, eval_samplesParameter
    
    if(seed != ""):
        print("Setting seed number to ", seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    else: print("Setting random seed")

    trainSubset = Subset(trainDataSet, range(train_samples))
    testSubset = Subset(testDataSet, range(test_samples))
    evalSubset = Subset(testDataSet, range(eval_samples))
    train_dataloader = DataLoader(trainSubset, batch_size=batch_size_training, shuffle=False)
    test_dataloader = DataLoader(testSubset, batch_size=batch_size_test, shuffle=False)
    eval_dataloader = DataLoader(evalSubset, batch_size=1, shuffle=False)
    print("Created all dataloaders")

def initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate):
    global model, criterion_class, chosen_optimizer, layers
    input_size = torch.flatten(train_dataloader.dataset[0][0]).numel()
    output_size = torch.flatten(train_dataloader.dataset[0][1]).numel()
    
    model = RENN.CustomizableRENN(input_size, hidden_sizes, output_size)
    model.to(device)
    layers = np.array(RENN.layers)
    
    if(loss_function == "MSE"):
        criterion_class = nn.MSELoss()  # For regression
    elif(loss_function == "Cross-Entropy"):
        criterion_class = nn.CrossEntropyLoss()  # For multi-class classification
    
    if(optimizer == "Adam"):
        chosen_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif(optimizer == "SGD"):
        chosen_optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train(epochs=10):
    global model, chosen_optimizer, criterion_class
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, classification in train_dataloader:
            images = images.float()
            images = images.to(device)
            classification = classification.float()
            classification = classification.to(device)
            chosen_optimizer.zero_grad()
            output = model(images)
            loss = criterion_class(output, classification)
            loss.backward()
            chosen_optimizer.step()
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

def trainModel(hidden_sizes, loss_function, optimizer, learning_rate, epochs):
    initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate)
    print("Model initialized, Starting training")
    train(epochs=epochs)
    print("Training finished")

def initializeHook(hidden_sizes, train_samples):
  global trainSubset

  hookDataLoader = DataLoader(trainSubset, batch_size=1, shuffle=False)
  RENN.initializeHook(hookDataLoader, model, hidden_sizes, train_samples)

"""# Evaluation: Output"""

import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

import plotly.graph_objects as go
import plotly.subplots as sp

def showIndividualImagesPlotly(images, layer, closestSources, showClosestMostUsedSources, mode):
    num_images = len(images)
    num_cols = 5  # Number of columns
    if num_images == 10:
        num_cols = 5
    num_rows = num_images // num_cols if (num_images // num_cols >= 1) else 1  # Number of rows
    if num_images % num_cols != 0:
        num_rows = num_images // num_cols + 1  # Number of rows

    fig = sp.make_subplots(rows=num_rows, cols=num_cols)

    for i, (image, sources, title) in enumerate(images):
        row_index = i // num_cols
        col_index = i % num_cols
        fig.add_trace(go.Image(name=f'<b>{title}</b><br>Closest {showClosestMostUsedSources} Sources compared by {mode}:<br>{sources}', z=image), row=row_index + 1, col=col_index + 1)

    fig.update_layout(
        title=f'Blended closest {closestSources} sources for each neuron in layer {layer} (compared to their {mode} output)',
        grid={'rows': num_rows, 'columns': num_cols},
        height=225 * num_rows,  # Adjust the height of the plot
        width=225 * num_cols,
        hoverlabel=dict(namelength=-1)
    )

    #fig.show()
    display(fig)

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
        axes[i+2].set_title(f"A - Source {source[0]} ({(blendedSourceImageActivation[1][i]*100.0):.2f}%): {image[1]}")
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
        axes[i+2].set_title(f"S - Source {source[0]} ({(blendedSourceImageSum[1][i]*100.0):.2f}%): {image[1]}")
        axes[i+2].imshow(Image.fromarray(image[0].cpu().numpy()*255))
    plt.show()

#Import dataclass package to create a class based approach instead of normal arrays
from dataclasses import dataclass
@dataclass(order=True)
class WeightedSource:
  source: int
  difference: float

def getMostUsedPerLayer(sources):
    mostUsed = []
    sourceCounter = 0
    for src in sources:
        mostUsed.append(src.source)
        sourceCounter += 1
    return sourceCounter, mostUsed

def getClosestSourcesPerNeuronAndLayer(sources, layersToCheck, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, mode=""):
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
                if not(visualizationChoice == "Custom" and ((cNeuron < int(visualizeCustom[cLayer][0][0])) or (cNeuron > int(visualizeCustom[cLayer][0][1])))):
                    imagesPerLayer.append([blendIndividualImagesTogether(weightedSourcesPerNeuron, closestSources), [f"Source: {source.source}, Difference: {source.difference:.10f}<br>" for source in weightedSourcesPerNeuron][:showClosestMostUsedSources], f"{mode} - Layer: {int(layersToCheck[cLayer]/2)}, Neuron: {cNeuron}"])

        if not(visualizationChoice == "Per Layer Only"):
            if not(mode == "Activation" and visualizationChoice == "Custom" and visualizeCustom[cLayer][1] == False):
                showIndividualImagesPlotly(imagesPerLayer, int(layersToCheck[cLayer]/2), closestSources, showClosestMostUsedSources, mode)

        if not(visualizationChoice == "Per Neuron Only"):
            if not(mode == "Activation" and visualizationChoice == "Custom" and visualizeCustom[cLayer][1] == False):
                weightedSourcesPerLayer = sorted(weightedSourcesPerLayer, key=lambda x: x.difference)
                sourceCounter, mostUsed = getMostUsedPerLayer(weightedSourcesPerLayer)
                counter = Counter(mostUsed)
                image = blendIndividualImagesTogether(counter.most_common()[:closestSources], closestSources, True)

                plt.figure(figsize=(28,28))
                plt.imshow(image)
                plt.title(f"{mode} - Layer:  {int(layersToCheck[cLayer]/2)}, {closestSources} most used Sources")
                plt.show()

"""# Evaluation: Visual Blending"""

def blendImagesTogether(mostUsedSources, mode):
    image = Image.fromarray(np.zeros(shape=[28,28], dtype=np.uint8)).convert("RGBA")
    weights = []
    total = 0

    if mode == "Not Weighted":
        for sourceNumber, counter in mostUsedSources:
            total += counter

        for sourceNumber, counter in mostUsedSources:
            image = Image.blend(image, Image.fromarray(x_train[sourceNumber].numpy()*255).convert("RGBA"), (counter/total))
            weights.append(counter/total)

    return (image, weights)

def blendIndividualImagesTogether(mostUsedSources, closestSources, layer=False):
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
    return ((array - min) / (max - min)) if (max-min) > 0.0 else np.zeros_like(array)

"""# Evaluation: Code"""

def visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom):
    global dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource
    #Make sure to set new dictionarys for the hooks to fill - they are global!
    dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource = RENN.initializeEvaluationHook(hidden_sizes, eval_dataloader, eval_samples, model)
    
    for pos, (sample, true) in enumerate(eval_dataloader):
        sample = sample.float()
        prediction = predict(sample)
    
        if(visualizationChoice == "Weighted"):
            sourcesSum, outputsSum, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], "Sum")
            mostUsedSourcesWithSum = RENN.getMostUsedSources(sourcesSum, closestSources, "Sum")
            #20 because otherwise the blending might not be visible anymore. Should be closestSources instead to be correct!
            blendedSourceImageSum = blendImagesTogether(mostUsedSourcesWithSum[:20], "Not Weighted")
    
            sourcesActivation, outputsActivation, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], "Activation")
            mostUsedSourcesWithActivation = RENN.getMostUsedSources(sourcesActivation, closestSources, "Activation")
            #20 because otherwise the blending might not be visible anymore. Should be closestSources instead to be correct!
            blendedSourceImageActivation = blendImagesTogether(mostUsedSourcesWithActivation[:20], "Not Weighted")
    
            showImagesUnweighted(createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedSourceImageActivation, blendedSourceImageSum, mostUsedSourcesWithActivation[:showClosestMostUsedSources], mostUsedSourcesWithSum[:showClosestMostUsedSources])
        else:
            sourcesSum, outputsSum, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], "Sum")
            mostUsedSourcesWithSum = getClosestSourcesPerNeuronAndLayer(sourcesSum, layerNumbersToCheck, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, "Sum")

            sourcesActivation, outputsActivation, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[pos], "Activation")
            mostUsedSourcesWithActivation = getClosestSourcesPerNeuronAndLayer(sourcesActivation, layerNumbersToCheck, closestSources, showClosestMostUsedSources, visualizationChoice, visualizeCustom, "Activation")
    
    #print(f"Time passed since start: {time_since_start(startTime)}")