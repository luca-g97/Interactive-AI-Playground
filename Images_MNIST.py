import torch
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import Customizable_RENN as RENN

mnist, to_categorical, nn, DataLoader, device = "", "", "", "", ""
train_dataloader, test_dataloader, eval_dataloader, x_train, y_train, x_test, y_test, x_eval, y_eval = "", "", "", "", "", "", "", "", ""
model, criterion_class, chosen_optimizer, layers = RENN.CustomizableRENN(10, [["", "", ""]], 10), "", "", ""
train_samples, eval_samples, test_samples = 1, 1, 1
dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource = [], []

def initializePackages(mnistPackage, to_categoricalPackage, nnPackage, DataLoaderPackage, devicePackage):
    global mnist, to_categorical, nn, DataLoader, device
    mnist, to_categorical, nn, DataLoader, device = mnistPackage, to_categoricalPackage, nnPackage, DataLoaderPackage, devicePackage

def createTrainAndTestSet():
    global trainSet, testSet, x_train, y_train, x_test, y_test, x_eval, y_eval
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
    
    trainSet = [(torch.flatten(x), torch.flatten(y)) for x, y in zip(x_train, y_train)]
    testSet = [(torch.flatten(x), torch.flatten(y)) for x, y in zip(x_test, y_test)]
    
    print(f"Created {len(trainSet)} Trainsamples & {len(testSet)} Testsamples")
    return trainSet, testSet

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

def initializeDatasets(train_samplesParameter, test_samplesParameter, eval_samplesParameter):
    global train_samples, test_samples, eval_samples
    global train_dataloader, test_dataloader, eval_dataloader, x_train, y_train, x_test, y_test, x_eval, y_eval
    train_samples, test_samples, eval_samples = train_samplesParameter, test_samplesParameter, eval_samplesParameter
    x_train, y_train = trainSet[0][:train_samples], trainSet[1][:train_samples]
    x_test, y_test = testSet[0][:test_samples], testSet[1][:test_samples]
    x_eval, y_eval = x_test[:eval_samples], y_test[:eval_samples]

    train_data = CustomMNISTData(mode="Train")
    test_data = CustomMNISTData(mode="Test")
    eval_data = CustomMNISTData(mode="Evaluate")

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False)

def initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate):
    global model, criterion_class, chosen_optimizer, layers
    input_size = len(torch.flatten(torch.tensor(train_dataloader.dataset[0][0])))
    output_size = len(torch.flatten(torch.tensor(train_dataloader.dataset[0][1])))
    
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

def train(model, criterion_class,  optimizer, epochs=10):
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
        #print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_dataloader)}, Validation Loss: {val_loss/len(test_dataloader)}')
def trainModel(hidden_sizes, loss_function, optimizer, learning_rate, epochs):
    initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate)
    train(model, criterion_class, chosen_optimizer, epochs=epochs)
    return model, train_dataloader

def initializeHook(hidden_sizes, train_samples):
  hookDataLoader = DataLoader(train_data, batch_size=1, shuffle=False)
  RENN.initializeHook(hookDataLoader, model, hidden_sizes, train_samples)

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

def visualize(eval_samples, closestSources, showClosestMostUsedSources, visualizationChoice):
    global dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource
    #Make sure to set new dictionarys for the hooks to fill - they are global!
    dictionaryForSourceLayerNeuron = np.zeros((eval_samples, len(layers)*2, np.max(layerSizes)))
    dictionaryForLayerNeuronSource = np.zeros((len(layers)*2, np.max(layerSizes), eval_samples))
    
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        attachHooks(eval_dataloader)
    
    for pos, (sample, true) in enumerate(eval_dataloader):
        sample = sample.float()
        prediction = predict(sample)
    
        if(visualizationChoice == "Weighted"):
            sourcesSum, outputsSum, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Sum")
    
            mostUsedSourcesWithSum = getMostUsedSources(sourcesSum, "Sum")
            blendedSourceImageSum = blendImagesTogether(mostUsedSourcesWithSum[:20], "Not Weighted")
    
            sourcesActivation, outputsActivation, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Activation")
            mostUsedSourcesWithActivation = getMostUsedSources(sourcesActivation, "Activation")
            blendedSourceImageActivation = blendImagesTogether(mostUsedSourcesWithActivation[:20], "Not Weighted")
    
            showImagesUnweighted(createImageWithPrediction(sample.reshape(28, 28), true, prediction), blendedSourceImageActivation, blendedSourceImageSum, mostUsedSourcesWithActivation[:showClosestMostUsedSources], mostUsedSourcesWithSum[:showClosestMostUsedSources])
        else:
            sourcesSum, outputsSum, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Sum")
            mostUsedSourcesWithSum = getClosestSourcesPerNeuronAndLayer(sourcesSum, layerNumbersToCheck, "Sum")
    
            sourcesActivation, outputsActivation, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, activationsByLayers, dictionaryForSourceLayerNeuron[pos], "Activation")
            mostUsedSourcesWithActivation = getClosestSourcesPerNeuronAndLayer(sourcesActivation, layerNumbersToCheck, "Activation")
    
    #print(f"Time passed since start: {time_since_start(startTime)}")