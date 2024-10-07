import torch
from torch import nn
import numpy as np
from collections import Counter
import LLM_Small1x1 as Small1x1

layer, source, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, activationsBySources, activationsByLayers, totalLayers = "", "", "", "", "", "", ""
llm, layerSizes, device, hidden_sizes, layers, layers, currentLayer, relevantLayerIndices = "", "", "", "", "", [], 0, []

def initializePackages(devicePackage, seed=""):
    global device, np, torch 
    device = devicePackage
    if(seed != ""):
        torch.manual_seed(seed)
        np.random.seed(seed)

def getLayer(hidden_layers, layerNumber, input_size=10, output_size=10):
    if(hidden_layers[layerNumber][0] == "Linear"):
        return nn.Linear(input_size, output_size)
    elif(hidden_layers[layerNumber][0] == "Conv2d"):
        return nn.Conv2d(input_size, output_size)
    return False

def getActivation(hidden_layers, layerNumber):
    if(hidden_layers[layerNumber][2] == "ReLU"):
        return nn.ReLU()
    elif(hidden_layers[layerNumber][2] == "Sigmoid"):
        return nn.Sigmoid()
    elif(hidden_layers[layerNumber][2] == "Tanh"):
        return nn.Tanh()
    return False

def checkIfActivationLayerExists(hidden_layers, layerNumber):
    if hidden_layers[layerNumber][2] != "None":
        return True
    return False

def createLayers(layerName, layerType, activationLayerType):
    global currentLayer, layers

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
        self.num_layers = (len(hidden_layers))
        self.hidden_layers = hidden_layers
        global layers, currentLayer, relevantLayerIndices

        layers = []
        currentLayer = 0
        relevantLayerIndices = []
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

            if (checkIfActivationLayerExists(self.hidden_layers, layer)):
                x = getattr(self, f'activation{layer}')(x)
        return x

# Forward hook
def forward_hook(module, input, output):
    global layer
    global source
    global dictionaryForSourceLayerNeuron
    global dictionaryForLayerNeuronSource
    global hidden_sizes
    global llm

    if not (isinstance(module, nn.Sequential) or isinstance(module, Small1x1.FeedForward) or isinstance(module, Small1x1.TransformerBlock) or isinstance(module, nn.Dropout)):
        if (llm == True):
            actualLayer = layer
            layerNeurons = layers[actualLayer][1]
            if(source >= dictionaryForSourceLayerNeuron.shape[0]):
                return
        else:
            actualLayer = int(layer/2)
            layerNeurons = layers[actualLayer][1].out_features

        correctTypes = False
        if(llm == False):
            activation_type = type(getActivation(hidden_sizes, actualLayer)) 
            layer_type = type(getLayer(hidden_sizes, actualLayer))
            if (type(module) == activation_type or type(module) == layer_type):
                correctTypes = True

        relevantOutput = output[0].cpu().numpy()

        #print(layer, layers[layer], relevantOutput.shape)

        if(correctTypes or llm):
            #Use for array structure like: [source, layer, neuron]
            if(len(relevantOutput.shape) > 1):
                if(relevantOutput.shape[1] != layerNeurons):
                    layerNeurons = relevantOutput.shape[1]
                    #layers[actualLayer] = (layers[actualLayer][0], relevantOutput.shape[1], layers[layer][2:])
            dictionaryForSourceLayerNeuron[source][layer,:layerNeurons] = relevantOutput
            # if(source == 0):
            #   print(relevantOutput, dictionaryForSourceLayerNeuron[source][layer,:layerNeurons])

            #Use for array structure like: [layer, neuron, source]
            output = relevantOutput if len(relevantOutput.shape) == 1 else relevantOutput[0]
            for neuronNumber, neuron in enumerate(output):
                if neuronNumber < layerNeurons:
                    dictionaryForLayerNeuronSource[layer][neuronNumber][source] = neuron
                else:
                    break

        if(layer % 2 == 0 and llm != True):
            if(checkIfActivationLayerExists(hidden_sizes, actualLayer)):
                layer += 1
            elif(layer == (len(layers)*2)-2):
                layer = 0
            else:
                layer += 2
        else:
            if((layer == (len(layers)*2)-1 and llm != True) or (layer == (len(layers))-1 and llm == True)):
                layer = 0
            else:
                layer += 1    

def attachHooks(hookLoader, model, llmType = False):
    global source, layer

    hooks = []  # Store the handles for each hook
    outputs = np.array([])

    for name, module in model.named_modules():
        if not isinstance(module, CustomizableRENN):
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)

    with torch.no_grad():
        # Forward Pass
        for source, (inputs, labels) in enumerate(hookLoader):
            layer = 0
            # Uncomment for array structure like: [source, layer, neuron]
            if not llmType:
                inputs = inputs.float()
            inputs = inputs.to(device)
            _ = model(inputs)

    # Remove hooks after use
    for hook in hooks:
        hook.remove()

def createDictionaries(hidden_sizes, totalLayersParameter, train_samples):
    global activationsBySources, activationsByLayers, totalLayers, layerSizes
    totalLayers = totalLayersParameter
    layerSizes = [size[1] for size in hidden_sizes[:]]
    activationsBySources = np.zeros((train_samples, totalLayers, np.max(layerSizes)))
    activationsByLayers = np.zeros((totalLayers, np.max(layerSizes), train_samples))
    print("Hook-Dictionaries created")# - ", "Activations by Sources (Shape): ", activationsBySources.shape, " | ", "Activations by Layers (Shape): ", activationsByLayers.shape)

def runHooks(train_dataloader, model, layersParameter=layers, llmType = False):
    global layers, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, activationsBySources, activationsByLayers, llm
    
    #Variables for usage within the hook
    llm = llmType
    layers = layersParameter
    dictionaryForSourceLayerNeuron = activationsBySources
    dictionaryForLayerNeuronSource = activationsByLayers

    attachHooks(train_dataloader, model, llmType)
    activationsBySources = dictionaryForSourceLayerNeuron
    activationsByLayers = dictionaryForLayerNeuronSource
    print("Hooks finished successfully")

def initializeHook(train_dataloader, model, hidden_sizesParameter, train_samples):
    global totalLayers, layer, hidden_sizes, source, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource, activationsBySources, activationsByLayers
    
    print("Initializing Hooks")
    hidden_sizes = hidden_sizesParameter
    totalLayers = len(layers)*2
    createDictionaries(hidden_sizes, totalLayers, train_samples)
    runHooks(train_dataloader, model, layers)

def initializeEvaluationHook(hidden_sizes, eval_dataloader, eval_samples, model):
    global dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource
    
    dictionaryForSourceLayerNeuron = np.zeros((eval_samples, totalLayers, np.max(layerSizes)))
    dictionaryForLayerNeuronSource = np.zeros((totalLayers, np.max(layerSizes), eval_samples))

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        attachHooks(eval_dataloader, model)

    return dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource

def identifyClosestSources(closestSources, outputs, mode = ""):
    global layers
    
    dictionary = activationsByLayers
    if(mode == "Sum"):
        layerNumbersToCheck = [idx*2 for idx, (name, layerNumber, activation) in enumerate(layers)]
    elif(mode == "Activation"):
        layerNumbersToCheck = [(idx*2)+1 for idx, (name, layerNumber, activation) in enumerate(layers) if getActivation(hidden_sizes, idx) != False]
    else:
        layerNumbersToCheck = [idx for idx, _ in enumerate(layers)]

    layersToCheck = dictionary[layerNumbersToCheck]
    outputsToCheck = outputs[layerNumbersToCheck]
    identifiedClosestSources = np.empty((len(layersToCheck), np.max(layerSizes), closestSources), dtype=tuple)

    #print(len(layersToCheck), len(outputsToCheck))
    for currentLayer, layer in enumerate(layersToCheck):
        for currentNeuron, neuron in enumerate(layer):
            maxNeurons = layers[currentLayer][1] if mode=="" else layers[currentLayer][1].out_features
            if(currentNeuron < maxNeurons):
                #print(currentLayer, currentNeuron, len(neuron), outputsToCheck[currentLayer][currentNeuron], outputsToCheck.shape)
                differencesBetweenSources = np.abs(neuron - np.full(len(neuron), outputsToCheck[currentLayer][currentNeuron]))
                sortedSourceIndices = np.argsort(differencesBetweenSources) # for highest difference uncomment this: [::-1]
                closestSourceIndices = sortedSourceIndices[:closestSources]
                tuples = tuple((closestSourceIndices[i], neuron[closestSourceIndices[i]], abs(neuron[closestSourceIndices[i]]-outputsToCheck[currentLayer][currentNeuron])) for i in range(closestSources))
                identifiedClosestSources[currentLayer][currentNeuron] = tuples
    return identifiedClosestSources, outputsToCheck, layerNumbersToCheck

def getMostUsed(sources, mode=""):
    mostUsed = []
    sourceCounter = 0
    for currentLayer, layer in enumerate(sources):
        for currentNeuron, neuron in enumerate(layer):
            maxNeurons = layers[currentLayer][1] if mode=="" else layers[currentLayer][1].out_features
            if(currentNeuron < maxNeurons):
                for sourceNumber, value, difference in neuron:
                    mostUsed.append(sourceNumber)
                    sourceCounter += 1
    return sourceCounter, mostUsed

def getMostUsedSources(sources, closestSources, weightedMode=""):
    weightedSources = []

    sourceCounter, mostUsed = getMostUsed(sources, weightedMode)
    counter = Counter(mostUsed)

    print("Total closest Sources :" , sourceCounter, " | ", closestSources, " closest Sources (",weightedMode,") in format: [SourceNumber, Occurances]: ", counter.most_common()[:closestSources])
    return counter.most_common()[:closestSources]
