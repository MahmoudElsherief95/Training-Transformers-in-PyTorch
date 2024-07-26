# Training-Transformers-in-PyTorch

In this assignment we will implement and train a small transformer model and compare it to the LSTM in the previous assignment.

## Exercise 1: Causal Self-Attention

<img width="1584" alt="image" src="https://github.com/user-attachments/assets/b42398f4-68e8-4f6c-9d72-cdeb247519e5">

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exercise 2: Multi-Head Attention

Write a class `MultiHeadCausalSelfAttention` that derives from `nn.Module` and extends the functionality of `CausalSelfAttention` from the previous exercise.
The `__init__` method takes arguments `hidden_size, n_head, dropout`. `n_head` specifies the number of attention heads and `dropout` specifies the intensity for the dropout layers.
The `forward` method should split the hidden dimension of the pre-activations (i.e., $Q, K, V$) in `n_head` equally sized parts and perform attention to these parts in parallel.
Apply the first dropout layer direcly after the softmax.
After the multiplication of the scores with the values, recombine the output of the distinct attention heads back into a single hidden dimension of size $D$, i.e., the resulting shape should be the shape of the input.
Then perform an additional output projection again resulting in a hidden dimension of $D$.
Finally, apply the second dropout layer after the output projection.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exercise 3: Multi-Layer Perceptron

Write a class `MLP` that derives from `nn.Module` and whose `__init__` method takes two arguments: `hidden_size` and `dropout`.
It should implement a 2-layer feedforward network with `hidden_size` inputs, `4*hidden_size` hiddens, and `hidden_size` outputs.
It should apply the GELU activation function to the hiddens and dropout to the outputs.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exercise 4: Block

Write a class `Block` that derives from `nn.Module` and whose `__init__` method takes arguments `hidden_size, n_head, dropout`.
It should apply `nn.LayerNorm`, `CausalMultiHeadSelfAttention`, `nn.LayerNorm`, `MLP` in that order and feature residual connections from the input to the output of `CausalMultiHeadSelfAttention` and from there to the output of `MLP`.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exercise 5: GPT

Write a class `GPT` that derives from `nn.Module` and whose `__init__` method takes arguments `vocab_size, context_size, hidden_size, n_layer, n_head, dropout`.
The `forward` method should take two arguments `x, y` representing sequences of input and target tokens, respectively, both of which have type `torch.long` and shape ($N$, $T$), and returns logits and loss as a tuple.
The `GPT` module should feature two `nn.Embedding` layers, one for token embeddings and one for positional embedding, i.e., it should embed the position of the corresponding token within the input sequence.
The positional embedding is necessary for the Transformer to determine the order of its inputs.
Add the two embeddings and apply a dropout layer.
Next, apply `n_layers` layers of `Block`s followed by a `nn.LayerNorm` and a `nn.Linear` (without bias) mapping to an output dimension of `vocab_size`.
Finally, apply the cross-entropy loss function to the logits.
To save some parameters, apply weight tying between the token embedding layer and the output layer, i.e., they should use the same weights.
Initialize all weights using a normal distribution with a mean of zero and a standard deviation of 0.02 (except for the output layers of the `MLP`s use $0.02/\sqrt{2 * \mathtt{n\_layer}}$) and all biases to zero.
Use the argument `dropout` as intensity for all dropout layers in the network.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exercise 6: Optimizer

Add a method `configure_optimizers` to the class `GPT` that takes arguments `weight_decay, learning_rate, betas`.
Divide the model parameters into two groups.
The first group consists of all parameters with at least 2 dimensions, e.g., weight/embedding matrices and uses a decay of `weight_decay`.
The second group consists of all other parameters, e.g., biases and layer norms, and does not use weight decay.
Construct and return a `torch.optim.AdamW` optimizer with `learning_rate` and `betas` that operates on these two parameter groups.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exercise 7: Training

In the code cell below you find some globals, helper functions, and boilerplate code. Extend the given code by a training loop that
* stops after `max_iters` iterations
* applies the learning rate schedule implemented in `get_lr`
* applies gradient clipping at `grad_clip` using `torch.nn.utils.clip_grad_norm_`
* accumulates gradients for `gradient_accumulation_steps` batches before each weight update
* logs the training loss and learning rate every `log_interval` iterations
* evaluates (and potentially checkpoints) the model using `estimate_loss` every `eval_iters` iterations.

The provided hyperparameter values should be a good guess for training a tiny model on CPU but feel free to experiment with them as you please. In particular, if you have a GPU available, you can try to scale things up a bit.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Exercise 8: Inference

Add a method `generate` to the class `GPT` that takes arguments `x, max_new_tokens, temperature=1.0`.
The method should take a batch of token sequences `x`, which it should extend by `max_new_tokens` new tokens generated by the model.
Once you have computed the logits for the next token, divide them by `temperature` before applying the softmax.
After applying the softmax, sample the next token from the resulting categorical distribution.
Try out different values for `temperature` and compare the results to those from the previous assignment.
