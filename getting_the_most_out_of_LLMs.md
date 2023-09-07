# Getting the most out of LLMs

Large Language Models (LLMs) such as GPT3/4, Falcon, and LLama are rapidly advancing in their ability to tackle human-centric tasks, establishing themselves as essential tools in modern knowledge-based industries.
Deploying these models in real-world tasks remains challenging, however:

-   In order to exhibit near-human text understanding and generation capabilities, LLMs currently require to be composed of billions of parameters (see [Kaplan et al](https://arxiv.org/abs/2001.08361), [Wei et. al](https://arxiv.org/abs/2206.07682)). This consequently amplifies the memory demands for inference.
-   In many real-world tasks, LLMs need to be given extensive contextual information. This necessitates the model's capability to manage very long input sequences during inference.

The crux of these challenges lies in augmenting the computational and memory capabilities of LLMs, especially when handling expansive input sequences.

In this blog post, we will go over the most effective techniques to tackle these challenges. for efficient LLM deployment:

1.  **Lower Precision**: Research has shown that operating at reduced numerical precision, namely 8bit and 4bit, can achieve computational advantages without a considerable decline in model performance.

2.  **Flash Attention:** Flash Attention is a variation of the attention algorithm that not only provides a more memory-efficient approach but also realizes increased efficiency due to optimized GPU memory utilization.

3.  **Architectural Innovations:** Considering that LLMs are always deployed in the same way during inference, namely autoregressive text generation with a long input context, specialized model architectures have been proposed that allow for more efficient inference. The most important advancement in model architectures hereby are [Alibi](https://arxiv.org/abs/2108.12409), [Rotary embeddings](https://arxiv.org/abs/2104.09864), [Multi-Query Attention (MQA)](https://arxiv.org/abs/1911.02150) and [Grouped-Query-Attention (GQA)]((https://arxiv.org/abs/2305.13245)).

Throughout this piece, we will offer an analysis of auto-regressive generation from a tensor's perspective, delve into the pros and cons of adopting lower precision, and provide a comprehensive exploration of the latest attention mechanisms and architectural developments. While doing so, we run practical examples showcasing each of the feature improvements explained above.
 
## 1. Harnessing the Power of Lower Precision {#1-harnessing-the-power-of-lower-precision}

Memory requirements of LLMs can be best understood by seeing the LLM as a set of weight matrices and vectors and the text inputs as a sequence of vectors. In the following, the definition *weights* will be used to signify all model weight matrices and vectors.

At the time of writing this paper, LLMs consist of at least a couple billion parameters meaning that the sum of all entries in all weights is larger than 1,000,000,000. Each entry thereby consists of a decimal number, e.g. `4.5689` which is usually stored in either [float32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format), [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format), or [float16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) format. This allows us to easily compute the memory requirement to load the LLM into memory:

> *Loading the weights of a model having X billion parameters requires roughly 4 * X GB of VRAM in float32 precision\*

Nowadays, models are however rarely trained in full float32 precision, but usually in bfloat16 precision or less frequently in float16 precision. Therefore the role of thumb becomes:

> *Loading the weights of a model having X billion parameters requires roughly 2 * X GB of VRAM in bfloat16/float16 precision\*

For shorter text inputs (less than 1024 tokens), the memory requirement for inference is very much dominated by the memory requirement to load the weights. Therefore, for now, let's assume that the memory requirement for inference is simply the memory requirement to load the model into the GPU VRAM.

To give some examples of how much VRAM it roughly takes to load a model in bfloat16:

-   **GPT3** requires 2 \* 175 GB = **350 GB** VRAM
-   [**Bloom**](https://huggingface.co/bigscience/bloom) requires 2 \* 176 GB = **352 GB** VRAM
-   [**Llama-2-70b**](https://huggingface.co/meta-llama/Llama-2-70b-hf) requires 2 \* 70 GB = **140 GB** VRAM
-   [**Falcon-40b**](https://huggingface.co/tiiuae/falcon-40b) requires 2 \* 40 GB = **80 GB** VRAM
-   [**MPT-30b**](https://huggingface.co/mosaicml/mpt-30b) requires 2 \* 30 GB = **60 GB** VRAM
-   [**bigcode/starcoder**](https://huggingface.co/bigcode/starcoder) requires 2 \* 15.5 = **31 GB** VRAM

As of writing this document, the largest GPU chip on the market is the A100 offering 80GB of VRAM. Most of the models listed before require more than 80GB just to be loaded and therefore necessarily require [tensor parallelism](https://huggingface.co/docs/transformers/v4.15.0/parallelism#tensor-parallelism) and/or [pipeline parallelism](https://huggingface.co/docs/transformers/v4.15.0/parallelism#naive-model-parallel-vertical-and-pipeline-parallel)

🤗 Transformers does not support tensor parallelism out of the box as it requires the model architecture to be written in a specific way. If you're interested in writing models in a tensor-parallelism-friendly way, feel free to have a look at [the text-generation-inference library](https://github.com/huggingface/text-generation-inference/tree/main/server/text_generation_server/models/custom_modeling).

Pipeline parallelism is supported out of the box. For this, simply load the model with `device="auto"` which will automatically place the different layers on the available GPUs.

If you are working with an 8 x 80GB A100 node, you could load BLOOM as follows

```bash
!pip install transformers accelerate bitsandbytes
```
```python
# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", device="auto")
```
 
which would equally distribute the attention layers automatically over all available GPUs.

Throughout this notebook, we will use [bigcode/octocoder](https://huggingface.co/bigcode/octocoder) as it can be run on a single 40 GB A100 GPU device chip. Note that all memory and speed optimizations that we will apply going forward, are equally applicable to models that require model or tensor parallelism.

Great, remembering our rule of thumb above we would expect the memory requirement to run inference with `bigcode/octocoder` to be around 31 GB VRAM. Let's give it a try.

We first load the model and tokenizer and then pass both to Transformers' [pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines).

The model is loaded in *bloat16* precision.
 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("bigcode/octocoder")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
```
 
```python
prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"

result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
result
```

Nice, we can now directly use the result to compute how many gigabytes were needed.
 
```python
def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024
```
 
```python
bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
```

**Output**:
```bash
    29.0260648727417
```
 
Close enough to our back-of-the-envelop computation! We can see the number is not exactly correct as going from bytes to kilobytes requires a multiplication of 1024 instead of 1000. Therefore the back-of-the-envelope formula can also be understood as an "at most X GB" computation.

Let's free some memory for the next experiments.
 
```python
del pipe
del model

import gc
import torch

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
```
 
```python
flush()
```
 
As we can see the memory requirement is roughly what we expected. Note that if we had tried to run the model in full float32 precision, a whopping 64 GB of VRAM would have been required.

> Almost all models are trained in bfloat16 nowadays, there is no reason to run the model in full float32 precision if [your GPU supports bfloat16](https://discuss.pytorch.org/t/bfloat16-native-support/117155/5). Float32 won't give better inference results than the precision that was used to train the model.

What if your GPU does not have 32 GB of VRAM? It has been found that model weights can be quantized to 8bit or 4bits without a significant loss in performance (see [Dettmers et al.](https://arxiv.org/abs/2208.07339)).

It is possible to quantize the models even further - to 3 or even 2 bits with an acceptable loss in performance as shown in the recent [GPTQ paper](https://huggingface.co/docs/transformers/main_classes/quantization#general-usage) 🤯.

Without going into too many details, quantization schemes aim at reducing the precision of weights while trying to keep the model's inference results as accurate as possible (*a.k.a* as close as possible to bfloat16).
Note that quantization works especially well for text generation since all we care about is choosing the *most likely next token* and don't really care about the exact values of the next token logit distribution. All that matters is that the next token logit distribution stays roughly the same so that an `argmax` or `topk` operation gives the same results.

There are various quantization techniques, which we won't discuss in detail here, but in general, all quantization techniques work as follows:

-   1.  Quantize all weights to the target precision

-   2.  Load the quantized weights, and pass the input sequence of vectors in bfloat16 precision

-   3.  Dynamically dequantize weights to bfloat16 to perform the computation with their input vectors in bfloat16 precision

-   4.  Quantize the weights again to the target precision after computation with their inputs.

In a nutshell, this means that input-matrix multiplications, with $X$ being the *inputs*, $W$ being a single-weight matrix and $Y$ is the output:

$Y = X * W$

are changed to

$ Y = X * \text{dequantize}(W); \text{quantize}(W) $

for every matrix multiplication.

Therefore, inference time is often **not** reduced when using quantized weights, but rather increases.
Enough theory, let's give it a try! In Transformers, you can very easily load weights as follows.
 
Make sure that the [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) library needs to be installed.
 
```bash
# !pip install bitsandbytes
```
 
and then we can load models in 8bit quantization by simply adding the `load_in_8bit` flag:
 
```python
model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", load_in_8bit=True, low_cpu_mem_usage=True)
```

Now, let's run our example again and measure the memory usage.

```python
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
result
```

**Output**:
```
    [{'generated_text': 'Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer: Here is a Python function that transforms bytes to Giga bytes:\n\n```python\ndef bytes_to_giga_bytes(bytes):\n    return bytes / 1024 / 1024 / 1024\n```\n\nThis function takes a single'}]
```

Nice, we're getting the same result as before, so no loss in accuracy! Let's look at how much memory was used this time.

```python
bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
```

**Output**:
```
    17.503968238830566
```

Significantly less, we're down to just 17.5 GBs and could therefore run this model on consumer GPUs like the 4090. Let's see what 4-bit quantization produces.

```python
del model
del pipe
```
 
```python
flush()
```

Nice, we're seeing a very nice gain in memory efficiency and more or less no degradation to the model's output. However, we can also notice a slight slow-down during inference.

4bit quantization can be loaded in the same way, by just passing `load_in_4bit=True` instead of `load_in_8bit=True`.


 
```python
model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", load_in_4bit=True, low_cpu_mem_usage=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
result
```

**Output**:
```
    [{'generated_text': 'Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer: Here                                                                               '}]
```

We see that this time the generate function prematurely stopped, showing a loss in accuracy! Let's see how much memory was required.

```python
bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
```

**Output**:
```
    9.543574333190918
```

```python
del model
del pipe
```
```python
flush()
```

Just 9.5GB! That's really not much for a \~16 billion parameter model, but we see that 4-bit quantization often comes at the cost of a loss in accuracy.

Running OctoCoder in 4bit reduces the required GPU VRAM from 32GB to just a bit over 9GB which is very low for a 15.5 billion parameter model. 4bit quantization allows the model to be run on GPUs such as RTX3090, V100, and T4 which are quite accessible for most people.

To quantize the model even further, we recommend looking into the [`AutoGPTQ`](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#autogptq-integration%60) implementation.

> Overall, it is important to remember that model quantization trades improved memory efficiency against accuracy and in most cases inference time.

If GPU memory is not a constraint for your use case, there is often no need to look into quantization. However many GPUs simply can't run LLMs without quantization methods and in this case, 4bit and 8bit quantization schemes are extremely useful tools.

For more in-detail usage information, we strongly recommend taking a look at the [Transformers Quantization Docs](https://huggingface.co/docs/transformers/main_classes/quantization#general-usage).

Next, let's look into how we can improve computational and memory efficiency by using better algorithms and an improved model architecture.


# 2. Flash Attention: A Leap Forward {#2-flash-attention-a-leap-forward}

Today's top-performing LLMs share more or less the same fundamental architecture that consists of feed-forward layers, activation layers, layer normalization layers, and most crucially, self-attention layers.

Self-attention layers are central to Large Language Models (LLMs) in that they enable the model to understand the contextual relationships between input words (or more specifically tokens).

> The catch however is that of all the layers self-attention layers grow \*quadratically( both in time and memory complexity with the sequence length $N$.

While this is not really noticeable for shorter input sequences of up to 1000 input tokens, it becomes a serious problem for longer input sequences of around 16000 input tokens.

Let's take a closer look. The formula to compute the output $\mathbf{O}$ of a self-attention layer for an input $\mathbf{X}$ of length $N$ is:

$$ \textbf{O} = \text{Attn}(\mathbf{X}) = \mathbf{V} \times \text{Softmax}(\mathbf{QK}^T) \text{ with } \mathbf{Q} = \mathbf{W}_q \mathbf{X}, \mathbf{V} = \mathbf{W}_v \mathbf{X}, \mathbf{K} = \mathbf{W}_k \mathbf{X} $$

$\mathbf{X} = (\mathbf{x}_1, ... \mathbf{x}_{N})$ is thereby the input sequence to the attention layer. The projections $\mathbf{Q}$ and $\mathbf{K}$ will therefore also each consist of $N$ vectors resulting in the $\mathbf{QK}^T$ being of size $N^2$.

LLMs usually have multiple attention heads, thus doing multiple such computations in parallel.
Assuming, the LLM has 80 attention heads and runs in bfloat16 precision, we can calculate the memory requirement to store the $\mathbf{QK^T}$ matrices to be $80 * 2 * N^2$ bytes. For $N=1000$ only around 0.1 GB of VRAM are needed, however, for $N=16000$ we would need 38 GB of VRAM, and for $N=100,000$ we would need \>1TB just to store the $\mathbf{QK}^T$ matrices.

Long story short, the default self-attention algorithm quickly becomes prohibitively memory-expensive for large input contexts.

As LLMs improve in text comprehension and generation, they are applied to increasingly complex tasks. While models once handled the translation or summarization of a few sentences, they now manage entire pages, demanding the capability to process extensive input lengths.

How can we get rid of the exorbitant memory requirements for large input lengths? We need a new way to compute the self-attention mechanism that gets rid of the $QK^T$ matrix. [Tri Dao et al.](FlashAttention:%20Fast%20and%20Memory-Efficient%20Exact%20Attention%20with%20IO-Awareness) developed exactly such a new algorithm and called it \*Flash Attention\*\*.

In a nutshell, Flash Attention breaks the \$ \\mathbf{V} \\times \\text{Softmax}(\\mathbf{QK}\^T) \$ computation apart and instead computes smaller chunks of the output by iterating oven multiple softmax computation steps:

$$ \textbf{O}_i \leftarrow s^a_{ij} * \textbf{O}_i + s^b_{ij} * \mathbf{V}_{j} \times \text{Softmax}(\mathbf{QK}^T_{i,j}) \text{ for multiple } i, j \text{ iterations} $$

with $s^a_{ij}$ and $s^b_{ij}$ being some softmax normalization statistics that need to be recomputed for every $i$ and $j$.

Please note that the whole Flash Attention is a bit more complex and is greatly simplified here as going in too much depth is out of scope for the document. The reader is invited to take a look at the well-written [Flash Attention paper](https://arxiv.org/pdf/2205.14135.pdf) for more details.

The main takeaway here is that:

> By keeping track of softmax normalization statistics and by using some smart mathematics, Flash Attention gives **numerical identical** outputs compared to the default self-attention layer at a memory cost that only increases linearly with $N$.

From looking at the formula, one would intuitively say that Flash Attention must be much slower compared to the default self-attention formula as more computation needs to be done. Indeed Flash Attention requires more FLOPs compared to normal attenion as the softmax normalization statistics have to constantly be recomputed.

> However Flash Attenion is much faster in inference compared to default attention which comes from its ability to significantly reduce the demands on the slower, high-bandwidth memory of the GPU (VRAM), focusing instead on the faster on-chip memory (SRAM).

Essentially, Flash Attention makes sure that all intermediate write and read operations can be done using the fast *on-chip* SRAM memory instead of having to access the VRAM memory.

Therefore there is absolutely no reason to **not** use Flash Attention if available.

Let's look at a practical example.

Our OctoCoder model now gets a significantly longer input prompt which includes a so-called system prompt\*. System prompts are used to steer the LLM into a better assistant tailored to the users' task. In the following, we use a system prompt that will make OctoCoder a better coding assistant.

```python
system_prompt = " Below are a series of dialogues between various people and an AI technical assistant.
The assistant tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble but knowledgeable.
The assistant is happy to help with code questions and will do their best to understand exactly what is needed.
It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer.
That said, the assistant is practical really does its best, and doesn't let caution get too much in the way of being useful.

The Starcoder models are a series of 15.5B parameter models trained on 80+ programming languages from The Stack (v1.2) (excluding opt-out requests).
The model uses Multi Query Attention, was trained using the Fill-in-the-Middle objective, and with 8,192 tokens context window for a trillion tokens of heavily deduplicated data.

-----

Question: Write a function that takes two lists and returns a list that has alternating elements from each input list.

Answer: Sure. Here is a function that does that.

def alternating(list1, list2):
   results = []
   for i in range(len(list1)):
       results.append(list1[i])
       results.append(list2[i])
   return results

Question: Can you write some test cases for this function?

Answer: Sure, here are some tests.

assert alternating([10, 20, 30], [1, 2, 3]) == [10, 1, 20, 2, 30, 3]
assert alternating([True, False], [4, 5]) == [True, 4, False, 5]
assert alternating([], []) == []

Question: Modify the function so that it returns all input elements when the lists have uneven length. The elements from the longer list should be at the end.

Answer: Here is the modified function.

def alternating(list1, list2):
   results = []
   for i in range(min(len(list1), len(list2))):
       results.append(list1[i])
       results.append(list2[i])
   if len(list1) > len(list2):
       results.extend(list1[i+1:])
   else:
       results.extend(list2[i+1:])
   return results

-----
""""
```

To achieve a critical long enough input sequence to showcase the memory gains of Flash Attention, we will simply duplicate the system prompt by 5 and add our coding question from above.

```python
long_prompt = 5 * system_prompt + prompt
```
 
Let's instantiate our model again in bfloat16 precision
 
```python
model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("bigcode/octocoder")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
```
 
and run it to gauge the memory requirement when using vanilla attention.

```python
import time

start_time = time.time()
result = pipe(long_prompt, max_new_tokens=60)[0]["generated_text"][len(long_prompt):]

print(f"Generated in {time.time() - start_time} seconds.")
result
```
For comparison, let's run the same function, but enable Flash Attention instead.

```python
# TODO: Wait for Flash Attn 2 support?
```

## 3. The Science Behind LLM Architectures: Strategic Selection for Long Text Inputs and Chat {#3-the-science-behind-llm-architectures-strategic-selection-for-long-text-inputs-and-chat}

So far we have looked into improving computational and memory efficiency by:

-   Casting the weights to a lower precision format
-   Improving the self-attention algorithm with more memory- and compute efficient algorithm

Let's now look into how we can change the architecture of an LLM so that it is most effective and efficient for:

-   Tasks requiring the LLM to handle long text inputs, such as retrieval augmented Questions Answering, Summarization, ...
-   Chat

Note that *Chat* also requires the model to handle long text inputs, but in addition, it necessitates that the model is able to efficiently handle the back-and-forth dialogue between user and assistant (such as ChatGPT).

Once trained, the fundamental LLM architecture is difficult to change, so it is important to make considerations about the LLM's tasks beforehand and accordingly optimize the model's architecture.

There are two important components of the model architecture that quickly become memory and/or performance bottlenecks for large input sequences and chat.

-   The positional embeddings
-   The key-value cache

Let's go over each component in more detail

### 3.1 Improving positional embeddings of LLMs {#31-improving-positional-embeddings-of-llms}

Self-attention puts each token in relation to each other's tokens.
As an example, the $\text{Softmax}(\mathbf{QK}^T)$ matrix of the text input sequence *"Hello", "I", "love", "you"* could look as follows:

![](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/self_attn_tokens.png)

Each word token is given a probability mass at which it attends all other word tokens and, therefore is put into relation with all other word tokens. E.g. the word *"love"* attends to the word *"Hello"* with 0.05%, to *"I"* with 0.3%, and to itself with 0.65%.

A LLM based on self-attention, but without position embeddings would have great difficulties in understanding the positions of the text inputs to each other. Due to self-attention, each word token appears to have the same distance from all others. This perception is based on the probability score computed by $\mathbf{QK}^T$ which relates each word token to each other word token in $O(1)$ computations regardless of their relative positional distance to each other. As a result, A LLM without positional embeddings would have a hard time understanding sentence order, *e.g.* differentiating between *"Hello I love you"* and *"You love I hello*.

To remedy this problem, the authors of [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) paper introduced sinusoidal positional embeddings $\mathbf{P} = \mathbf{p}_1, \ldots, \mathbf{p}_N$ that were simply added to the input sequence $\mathbf{\hat{X}} = \mathbf{\hat{x}}_1, \ldots, \mathbf{\hat{x}}_N$ = \$\\mathbf{x}\_1 + \\mathbf{p}\_1, \\ldots, \\mathbf{x}\_N + \\mathbf{x}\_N \$. Each $\mathbf{p}_i$ therefore has a unique signature that purely depends on its position $i$ and therefore can cue the model to better learn sentence order.

Instead of using fixed sinusoidal position embeddings, other work such as [Devlin et al.](https://arxiv.org/abs/1810.04805) used learned embeddings to let the model learn the positional embeddings $P$.

Sinoidal and learned position embeddings were the predominant methods to encode sentence order into LLMs, but problems have been found:

-   1.) Sinoidal and learned position embeddings are both absolute positional embeddings, *i.e.* encoding a unique embedding for each position id: $0, \ldots, N$. As has been found by [Huang et al.](https://arxiv.org/abs/2009.13658) and [Su et al.](https://arxiv.org/abs/2104.09864)\], absolute positional embeddings lead to poor LLM performance for long text inputs. For long text inputs, it is advantageous if the model learns the relative positional distance input text tokens have to each other instead of their absolute position.

-   2.) Learned position embeddings have to be trained on a fixed input length $N$, making it difficult to extrapolate to an input length longer than what it was trained on.

Recently, relative positional embeddings that can tackle the problems mentioned before have become more popular, most notably:

-   [Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864)
-   [ALiBi](https://arxiv.org/abs/2108.12409)

Both *RoPE* and *ALiBi* argue that it's best to cue the LLM about sentence order directly in the self-attention algorithm as it's there that word tokens are put into relation with each other. More specifically, sentence order should be cued by modifying the $\mathbf{QK}^T$ computation.

Without going into too many details, *RoPE* notes that positional information can be encoded into query-key pairs, *e.g.* $\mathbf{q}_i$ and $\mathbf{x}_j$ by rotating each vector by an angle $\theta * i$ and $\theta * j$ respectively with $i, j$ describing each vectors positional id:

$$ \mathbf{\hat{q}}_i^T \mathbf{\hat{x}}_j = \mathbf{{q}}_i^T \mathbf{R}_{\theta, i -j} \mathbf{{x}}_j, $$

with $R_{\theta, i - j}$ being a rotational matrix. $\theta$ is *not* learned during training, but instead set to a pre-defined value that depends on the maximum input sequence length during training.

> By doing so, the propability score between $\mathbf{q}_i$ and $\mathbf{q}_j$ is only affected if $i \ne j$ and is only affected by relative distance $i - j$ regardless of each vector's specific positions $i$ and $j$.

*RoPE* is used in multiple of today's most important LLMs, such as:

-   [**Falcon**](https://huggingface.co/tiiuae/falcon-40b)
-   [**Llama**](https://arxiv.org/abs/2302.13971)
-   [**PaLM**](https://arxiv.org/pdf/2204.02311.pdf)

As an alternative, *ALiBi* proposes a much simpler relative position encoding scheme. The relative distance input tokens have to each other is added as a negative integer scaled by `m` to each query-key entry of the $\mathbf{QK}^T$ matrix right before the softmax computation.

![](https://raw.githubusercontent.com/patrickvonplaten/scientific_images/master/alibi.png)

$m$ is *not* learned during training but instead set to a pre-defined value.
As shown in the [ALiBi](https://arxiv.org/abs/2108.12409) paper, this simple approach to relative positional encoding works very well in practice and scales well to long text input sequences.

*ALiBi* is used in multiple of today's most important LLMs, such as:

-   [**MPT**](https://huggingface.co/mosaicml/mpt-30b)
-   [**BLOOM**](https://huggingface.co/bigscience/bloom)

Both *RoPE* and *ALiBi* position encodings can extrapolate to input lengths not seen during training where it has been shown to work much better out-of-the-box for ALiBi as compared to *RoPE*.
For ALiBi, one simply increases the values of the lower triangular position matrix to match the length of the input sequence.
For *RoPE*, it has been shown that using the same $\theta$ that was used during training leads to poor results when passing text inputs much longer than those seen during training, *c.f* [Press et al.](https://arxiv.org/abs/2108.12409). However, the community has found a couple of effective tricks that adapt $\theta$ that allow *RoPE* position embeddings to be extrapolated (see [here](https://github.com/huggingface/transformers/pull/24653)).

> Both RoPE and ALiBi are relative positional embeddings that are *not* learned during training, but instead are based on the following intuitions:

-   Positional cues about the text inputs should be given directly to the $QK^T$ matrix of the self-attention layer
-   The LLM should be incentivized to learn a constant *relative* distance positional encodings have to each other
-   The further text input tokens are from each other, the lower the probability of their query-value probability. Both RoPE and ALiBi lower the query-key probability of tokens far away from each other. RoPE by decreasing their vector product by increasing the angle between the query-key vectors. ALiBi by adding large negative numbers to the vector product

In conclusion, LLMs that are intended to be deployed in tasks that require handling large text inputs are better trained with relative positional embeddings, such as RoPE and ALiBi. Also note that even if an LLM with RoPE and ALiBi has been trained only on a fixed length of say $N_1 = 2048$ it can still be used in practice with text inputs much larger than $N_1$, like $N_2 = 8192 > N_1$ by extrapolating the positional embeddings.

### 3.2 The key-value cache

Auto-regressive text generation with LLMs works by iteratively putting in an input sequence, sampling the next token, appending the next token to the input sequence, and continuing to do so until the LLM produces a token that signifies that the generation has finished.

Please have a look at [Transformer's Generate Text Tutorial](https://huggingface.co/docs/transformers/llm_tutorial#generate-text) to get a more visual explanation of how auto-regressive generation works.

Let's run a quick code snippet to show how auto-regressive works in practice. We will simply take the most likely next token via `torch.argmax`.

```python
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

for _ in range(5):
  next_logits = model(input_ids)["logits"][:, -1]
  next_token_id = torch.argmax(next_logits)

  input_ids = torch.cat([input_ids, next_token_id])
  print("shape of input_ids", input_ids.shape)

generated_text = tokenizer.batch_decode(input_ids[:, -5:])
generated_text
```
 
As we can see every time we increase the text input tokens by the just sampled token.

With very few exceptions, LLMs are trained using the [causal language modeling objective](https://huggingface.co/docs/transformers/tasks/language_modeling#causal-language-modeling) and therefore mask the upper triangle matrix of the attention score - this is why in the two diagrams above the attention scores are left blank (*a.k.a* have 0 probability). For a quick recap on causal language modeling you can refer to the [*Illustrated Self Attention blog*](https://jalammar.github.io/illustrated-gpt2/#part-2-illustrated-self-attention).

As a consequence, tokens *never* depend on previous tokens, more specifically the $\mathbf{q}_i$ vector is never put in relation with any key, values vectors $\mathbf{k}_j, \mathbf{v}_j$ if $j > i$. Instead $\mathbf{q}_i$ only attends to previous key-value vectors $\mathbf{k}_{m < i}, \mathbf{v}_{m < i} \text{ , for } m \in \{0, \ldots i - 1\}$. In order to reduce unnecessary computation, one can therefore cache the k/v vectors of all previous timesteps and all previous layers.

In Transformers, we can easily retrieve the key-value cache by passing the `use_cache` flag to the `forward` call.

```python
past_key_values = None # past_key_values is the key-value cache
generated_tokens = []

for _ in range(5):
  next_logits, past_key_values = model(input_ids, past_key_values=past_key_values, use_cache=True).to_tuple()
  input_ids = torch.argmax(next_logits)

  print("shape of input_ids", input_ids.shape)
  print("length of key-value cache", len(past_key_values[0][0][1]))  # past_key_values are of shape [num_layers, 0 for k, 1 for v, batch_size, length, hidden_dim]
  generated_tokens.append(input_ids)

generated_text = tokenizer.batch_decode(generated_tokens)
generated_text
```

This time, the text input tokens are *not* increased, but stay constant. Instead, every time the length of the key-value cache is increased by one corresponding to the new cached key-value states of every layer.

> Making use of the key-value cache means that the $\mathbf{QK}^T$ is essentially reduced to $\mathbf{q}_c\mathbf{K}^T$ with $\mathbf{q}_c$ being the query projection of the currently passed input token which is *always* just a single vector.

This has two advantages:

-   Significant increase in computational efficiency as much fewer computations are done compared to computing the full $\mathbf{QK}^T$ matrix. This leads to a big increase in inference speed
-   The maximum required memory is not increased quadratically with the number of generated tokens, but only linearly.

> In short, one should *always* make use of the key-value cache as it leads to identical results and a significant speed-up for longer input sequences. Transformers has the key-value cache enabled by default when making use of the text pipeline or the [`generate` method](https://huggingface.co/docs/transformers/main_classes/text_generation).

The key-value cache is especially useful for applications such as chat where multiple passes of auto-regressive decoding are required

```
User: How many people live in France?
Assistant: Roughly 75 million people live in France
User: And how many are in Germany?
Assistant: Germany has ca. 81 million inhabitants
```

In this chat, the LLM runs auto-regressive decoding twice:
- 1. The first time, the key-value cache is empty and the input prompt is `"User: How many people live in France?"` and the model auto-regressively generates the text `"Roughly 75 million people live in France"` while increasing the key-value cache at every decoding step.
- 2. The second time the input prompt is `"User: How many people live in France? \n Assistant: Roughly 75 million people live in France \n User: And how many in Germany?"`. But thanks to the key-value cache, all key-value vectors for the first two sentences are already computed therefore the input prompt only needs to be `"User: And how many in Germany?"`. The key-value vectors of the input prompt are then concatenated to the key-value cache of the first step and the second Assistant's answer `"Germany has ca. 81 million inhabitants"` autoregressive decoding.

Two things should be noted here:
- 1. Keeping all the context is crucial for LLMs deployed in chat so that the LLM knows e.g. that the user refers to the population when asking `"And how many are in Germany"`.
- 2. The key-value cache is extremely useful for chat as it allows us to continuously grow the encoded chat history instead of having to re-encode the chat history again from scratch (as e.g. would be the case when using an encoder-decoder architecture).

There is however one catch. While the required peak memory for the $\mathbf{QK}^T$ matrix is significantly reduced, just holding the key-value cache in memory becomes also very expensive, the longer the input sequence. Remember that the key-value cache needs to hold the key-value vectors for all previous input vectors $\mathbf{x}_i \text{, for } i \in \{1, \ldots, c - 1\}$ for all self-attention layers and for all attention heads.

Let's compute the number of float values that need to be stored in the key-value cache for our LLM.

$$ 2 \times (\text{seq_len} - 1) \times \text{num_attn_heads} \times \text{attn_head_dim} \times \text{num_layers} $$

Computing this for our LLM at a hypothetical input sequence length of 16000 gives:

```python
config = model.config
2 * 16_000 * config.n_layer * config.n_head * config.n_embd // config.n_head
```

Roughly 8 billion float values! Storing 8 billion float values in float16 precision requires roughly 15 GB of RAM roughly half as much as the model weights themselves!

Researchers have proposed two methods that allow to significantly reduce the memory cost of storing the key-value cache:

-   1.  [Multi-Query-Attention (MQA)](https://arxiv.org/abs/1911.02150)

Multi-Query-Attention was proposed in Noam Shazeer's *Fast Transformer Decoding: One Write-Head is All You Need* paper. As the title says, Noam found out that instead of using `n_head` or key-value projections weights, one can just use a single head-value projection weight pair that is shared across all attention heads without that the model's performance significantly degrades. 

> By using a single head-value projection weight pair, the key value vectors $\mathbf{k}_i, \mathbf{v}_i$ have to be identical across all attention heads which in turn means that we only need to store 1 key-value projection pair in the cache instead of `n_head` ones.

As most LLMs use between 20 and 100 attention heads, MQA significantly reduces the memory consumption of the key-value cache. For the LLM used in this notebook we could therefore reduce the required memory consumption from 15 GB to less than 400 MB at an input sequence length of 16_000.

In addition to memory savings, MQA also leads to improved computational efficiency as explained in the following.
In auto-regressive decoding, large key-value vectors need to be reloaded, concatenated with the current key-value vector pair to be then fed into the $\mathbf{q}_c\mathbf{K}^T$ computation at every step. For auto-regressive decoding, the required memory bandwidth for the constant reloading can become a serious time bottleneck. By reducing the size of the key-value vectors less memory needs to be accessed, thus reducing the memory bandwidth bottleneck.

The important part to understand here is that reducing the number of key-value attention heads to 1 only makes sense if a key-value cache is used. The peak memory consumption of the model for a single forward pass without key-value cache stays unchanged as every attention head still has a unique query vector so that each attention head still has a different $\mathbf{QK}^T$ matrice.

MQA has seen wide adoption by the community and is now used by many of the most popular LLMs:

-   [**Falcon**](https://huggingface.co/tiiuae/falcon-40b)
-   [**PaLM**](https://arxiv.org/pdf/2204.02311.pdf)
-   [**MPT**](https://huggingface.co/mosaicml/mpt-30b)
-   [**BLOOM**](https://huggingface.co/bigscience/bloom)

Also, the checkpoint used in this notebook - `bigcode/octocoder` - makes use of MQA.

-   2.  [Grouped-Query-Attention (GQA)](https://arxiv.org/abs/2305.13245)

Grouped-Query-Attention, as proposed by Ainslie et al. from Google, found that using MQA can often lead to quality degradation compared to using vanilla multi-key-value head projections. The paper argues that more model performance can be kept by less drastically reducing the number of query head projection weights. Instead of using just a single key-value projection weight, `n < n_head` key-value projection weights should be used. By choosing `n` to a significantly smaller value than `n_head`, such as 2,4 or 8 almost all of the memory and speed gains from MQA can be achieved while sacrificing less model capacity and thus arguably less performance.

Moreover, the authors of GQA found out that existing model checkpoints can be *uptrained* to have a GQA architecture with as little as 5% of the original pre-training compute. While 5% of the original pre-training compute can still be a massive amount, GQA *uptraining* allows existing checkpoints to be useful for longer input sequences.

GQA was only recently proposed which is why there is less adoption at the time of writing this blog post.
The most notable application of GQA is [Llama-v2](https://huggingface.co/meta-llama/Llama-2-70b-hf).

> As a conclusion, it is strongly recommended to make use of either GQA or MQA if the LLM is deployed with auto-regressive decoding and is required to handle large input sequences as is the case for chat *e.g.*.
 
## Conclusion

The research community is constantly coming up with new, nifty ways to speed up inference time for ever-larger LLMs. As an example, one such promising research direction is [speculative decoding](https://arxiv.org/abs/2211.17192) where "easy tokens" are generated by smaller, faster models and only "hard tokens" are generated by the LLM itself. Going into more detail is out of the scope of this notebook, but can be read in this [nice blog post](https://huggingface.co/blog/assisted-generation).

The reason massive LLMs such as GPT3/4, Llama-2-70b, Claude, PaLM can run so quickly in applications such as [Hugging Face Chat](https://huggingface.co/chat/) or ChatGPT is to a big part thanks to the above-mentioned improvements in precision, algorithms, and architecture. To get a better intuitive understanding of how the improvements function, we strongly recommend playing around with the above code snippets and trying to apply them in different real-world scenarios, such as *chat*, *long-context question answering*, etc...

Going forward, accelerators such as GPUs, TPUs, etc... will only get faster and allow for more memory, but one should nevertheless always make sure to use the best available algorithms and architectures to get the most bang for your
buck 🤗