import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
import Customizable_RENN as RENN

random, lorem, device, tiktoken, DataLoader, writer = "", "", "", "", "", ""
train_samples, test_samples, eval_samples = "", "", ""
GPT_CONFIG_124M, settings = "", ""
train_loader, val_loader, eval_loader, tokenizer  = "", "", "", ""
dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource = [], []
small1x1 = []

def initializePackages(randomPackage, loremPackage, devicePackage, tiktokenPackage, DataLoaderPackage, writerPackage):
    global random, lorem, device, tiktoken, DataLoader, writer
    random, lorem, device, tiktoken, DataLoader, writer = randomPackage, loremPackage, devicePackage, tiktokenPackage, DataLoaderPackage, writerPackage

def createUniqueCalculation(createdCalculations, xMin = 0, xMax = 9, yMin = 0, yMax = 9):
    x = random.randint(xMin, xMax)
    y = random.randint(yMin, yMax)
    while (x, y) in createdCalculations:
        x = random.randint(xMin, xMax)
        y = random.randint(yMin, yMax)
    createdCalculations.append((x, y))
    return createdCalculations, x, y

def createTrainAndTestSet(samples):
    global small1x1
    createdCalculations = []
    small1x1 = []

    for trainExample in range(samples):
        createdCalculations, x, y = createUniqueCalculation(createdCalculations, 0, 9, 0, 9)
        small1x1.append((x, y, x*y, lorem.sentence() + f"{x}*{y}={x*y}"))
    print(f"Created {len(small1x1)} samples for Small1x1")
    return small1x1

def setGPTSettings(layerAmount, learningRate, epochs):
    global GPT_CONFIG_124M, settings
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,   # Vocabulary size
        "context_length": 256, # Shortened context length (orig: 1024)
        "emb_dim": 768,        # Embedding dimension
        "n_heads": layerAmount,         # Number of attention heads
        "n_layers": layerAmount,        # Number of layers
        "drop_rate": 0.1,      # Dropout rate
        "qkv_bias": False      # Query-key-value bias
    }

    settings = {"learning_rate": learningRate, "weight_decay": 0.1, "batch_size": 1, "num_epochs": epochs}

    LLM_Layers = [[('Embedding', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["vocab_size"]),
     ('Embedding', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"])],
     #('Dropout', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["drop_rate"])],
      [('LayerNorm', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
     ('Linear', GPT_CONFIG_124M["vocab_size"], GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["vocab_size"])]]

    TransformerBlockLayer = [('LayerNorm', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
     ('Linear', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
     ('Linear', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
     ('Linear', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
     #('Dropout', 2, GPT_CONFIG_124M["drop_rate"]),
     ('Linear', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"]),
     ('MultiHeadAttention', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["emb_dim"])]
     #('Dropout', GPT_CONFIG_124M["emb_dim"], GPT_CONFIG_124M["drop_rate"])]
    
    return LLM_Layers, TransformerBlockLayer

"""#Data Initialization"""

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)

    return dataloader

def createLLMLoaders(train_samplesParameter, test_samplesParameter, eval_samplesParameter):
    global train_loader, val_loader, eval_loader, train_samples, test_samples, eval_samples
    train_samples, test_samples, eval_samples = train_samplesParameter, test_samplesParameter, eval_samplesParameter

    actualSamples = [sample[-1] for sample in small1x1]
    samples = "\n".join(actualSamples[:train_samples])

    train_loader = create_dataloader_v1(
        samples,
        batch_size=settings["batch_size"],
        max_length=1,
        stride=1,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    samples = "\n".join(actualSamples[train_samples:train_samples+test_samples])
    val_loader = create_dataloader_v1(
        samples,
        batch_size=settings["batch_size"],
        max_length=1,
        stride=1,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    samples = "\n".join(actualSamples[train_samples+test_samples:train_samples+test_samples+eval_samples])
    eval_loader = create_dataloader_v1(
        samples,
        batch_size=settings["batch_size"],
        max_length=1,
        stride=1,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

"""# Setup Customizable Network"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        in_idx.to(device)
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def initializeDatasets(train_samples, test_samples, eval_samples, seed=""):
    global torch

    if(seed != ""):
        torch.manual_seed(seed)

    createLLMLoaders(train_samples, test_samples, eval_samples)

def initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate):
    global model, criterion_class, chosen_optimizer, layers
    input_size = len(torch.flatten(torch.tensor(train_loader.dataset[0][0])))
    output_size = len(torch.flatten(torch.tensor(train_loader.dataset[0][1])))

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    layers = hidden_sizes

    if(loss_function == "MSE"):
        criterion_class = nn.MSELoss()  # For regression
    elif(loss_function == "Cross-Entropy"):
        criterion_class = nn.CrossEntropyLoss()  # For multi-class classification

    if(optimizer == "Adam"):
        chosen_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif(optimizer == "SGD"):
        chosen_optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, num_batches=None):
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        print("Dataloader: ", num_batches, ", ", len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch)
            total_loss += loss.item()
        else:
            break

    return total_loss / max(1, num_batches)

def train_model_simple(train_loader, val_loader, optimizer, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous epoch
            loss = calc_loss_batch(input_batch, target_batch)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            #if global_step % eval_freq == 0:
        model.eval()
        with torch.no_grad():
            train_loss, val_loss = evaluate_model(
                train_loader, val_loader, eval_iter)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
        #print(f"Ep {epoch+1} (Step {global_step:06d}): "f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        print(f"Ep {epoch+1}: "f"Train loss {train_loss}, Val loss {val_loss}")


    return train_losses, val_losses, track_tokens_seen

def evaluate_model(train_loader, val_loader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()

def main():
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        train_loader, val_loader, optimizer,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="=", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, train_loader, eval_loader

def trainModel(hidden_sizes, loss_function, optimizer, learning_rate, epochs):
    global train_loader, eval_loader, tokenizer
    initializeTraining(hidden_sizes, loss_function, optimizer, learning_rate)
    _, _, _, train_dataloader, eval_dataloader = main()
    tokenizer = tiktoken.get_encoding("gpt2")

def initializeHook(hidden_sizes, train_samples):
    RENN.createDictionaries(hidden_sizes, len(hidden_sizes), train_samples)
    RENN.runHooks(train_loader, model, hidden_sizes, True)

def generate(model, idx, max_new_tokens, context_size, temperature, top_k=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond.to(device))
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx.to(device), idx_next.to(device)), dim=1)  # (batch_size, num_tokens+1)

    return idx

def getLLMPrediction(sample):
    x, y, solution, _ = sample
    index = text_to_token_ids(f"{x}*{y}=", tokenizer)
    maxNewTokens = len(str(x*y))
    contextSize = len(str(f"{x}*{y}="))
    token_ids = generate(model=model, idx=index, max_new_tokens=maxNewTokens,
    context_size=contextSize, top_k=1, temperature=1.0)
    prediction = token_ids_to_text(token_ids, tokenizer)
    return x, y, solution, prediction

def visualize(hidden_sizes, closestSources, showClosestMostUsedSources, visualizationChoice):
    global train_samples, test_samples, eval_samples, dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource

    dictionaryForSourceLayerNeuron, dictionaryForLayerNeuronSource = RENN.initializeEvaluationHook(hidden_sizes, eval_loader, eval_samples, model)
    
    for sampleNumber in range(eval_samples):
        #TODO: Save calculations to file and hook in evaluation mode onto the model!
        sources, outputs, layerNumbersToCheck = RENN.identifyClosestSources(closestSources, dictionaryForSourceLayerNeuron[sampleNumber])
        #print(activationsByLayers, dictionaryForSourceLayerNeuron[sampleNumber])
        mostUsedSources = RENN.getMostUsedSources(sources, closestSources, "")
        x, y, solution, prediction = getLLMPrediction(small1x1[train_samples+test_samples+sampleNumber])
        print(prediction, f" -> Difference = {(solution) - (int(prediction) if prediction.isdigit() else 0)}")
        print([(sourceNumber, count, small1x1[sourceNumber][3]) for sourceNumber, count in mostUsedSources])
    
    #print(f"Time passed since start: {time_since_start(startTime)}")