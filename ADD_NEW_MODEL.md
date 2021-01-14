# How to add [name of model] to 🤗Transformers
	
Teacher: [name of hugging face teacher]
	
Begin: [start date]
	
Estimated time: [end date]

The following sections explain in detail how to add [name of model] 
to 🤗Transformers. You will work closely with [name of hugging face teacher] to
integrate [name of model] into 🤗Transformers. By doing so, you will both gain a 
theoratical and deep practical understanding of [name of model]. But more importantly, 
you will have made a major open-source contribution to 🤗Transformers. Along the way,
you will:
	
- get insights into open-source best practices, 
- understand the design principles of one of the most popular NLP libraries,
- learn how to do efficiently test large NLP models,
- learn how to integrate python utilities like `black`, `isort`, `make fix-copies` into a library 
  to always ensure clean and readable code.

To begin with, you should start by getter a good understanding of the model.
  
## Theoritacl aspects of [name of model]
  
### Paper

  You should take some time to read [[name of model]'s paper]([link to paper]).
  There might be large sections of the paper that are difficult to understand.
  If this is the case, this is totally fine - don't worry! The goal is not to get 
  a deep theoretical understanding of the paper, but to extract the necessary information 
  required to effectively reimplement the model to 🤗Transformers.
  That being said, you don't have to spend too much time on the theoretical aspects,
  but rather focus on the practical ones, namely:
  
  - What time of model is [name of model]? BERT-like encoder-only model? GPT2-like decoder-only model? BART-like encoder-decoder model?
  - What are the applications of [name of model]? Text classification? Text generation? Seq2Seq tasks, *e.g.* summarization?
  - What is the novel feature of the model making it different to BERT or BART (if it's an encoder-decoder model)?
  - Which of the already existing [🤗Transformers models](https://huggingface.co/transformers/#contents) is most similar to [name of model]?
  
 After you feel like you have gotten a good overview over the architecture of the model, you might want 
 to ping [name of hugging face teacher] for any questions you might have.
 This might include questions regarding the model's architecture, its attention layer, etc. [name of hugging face teacher] will be more 
 than happy to help you.
 
### Additional resources

 Before diving into the code, here are some additional resources that might be worth taking a look at:
 
 - [link 1]
 - [link 2]
 - [link 3]
 - ...
 
### Make sure you've understood the fundamental aspects of [name of model]
 
Alright, now you should be ready to take a closer look into the actualy code of [name of model].
You should have understood the following aspects of [name of model] by now:

- [characteristic 1 of name of model]
- [characteristic 2 of name of model]
- ...

If any of the mentioned aspects above are **not** clear to you, now is a great time to talk to [name of hugging face teacher] again!

## Next prepare your environment

1. Fork the [repository](https://github.com/huggingface/transformers) by clicking on the 'Fork' button on the repository's page.
This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

	```bash
	git clone https://github.com/[your Github handle]/transformers.git
	cd transformers
	git remote add upstream https://github.com/huggingface/transformers.git
	```
 
3. Set up a development environment, for instance by running the following command:

	```bash
	conda create -n env python=3.7 --y
	conda activate env
	pip install -e ".[dev]"
	```
  
  and return to the parent directory
  
  ```bash
  cd ..
  ```

4. We recommend to add the PyTorch version of [name of model] to Transformers. In order to install PyTorch,
  please follow the instructions on: https://pytorch.org/. 
  
  **Note:** You don't need to have CUDA install. It is sufficient to just be working on CPU.

5. To port [name of model], you will also need access to its [original repository]([link to original repo]):
  
  ```bash
  git clone [clone link to original repo]
  cd [name of repo]
  pip install -e .
  ```

Now you have set up a development environment to port [name of model] to 🤗Transformers.
  
## Run a pre-trained checkpoint using the original repository

At first, you will work the original repository. Often, the original implementation is very
 "researchy" meaning that documentation might be lacking and the code can be hard to read / understand.
But this should be exactly your motivation to reimplement [name of the model]. At Hugging Face, one of our main goals is to
*make people stand on the shoulders of giants* which translates here very well into taking a working 
model and rewriting it to make it as **accesable, user-friendly, and beautiful** as possible.
This is the #1 motivation to reimplement models into 🤗Transformers - trying to maximize access 
to a complex new NLP technology for **everybody**.
	
You should start thereby by diving into the original repository.

### Get familiar with the original repository.

Succesfully running the official pre-trained model in the original repository is often 
**the most difficult** step. From our experience, it is very important to spend some time to 
get familiar with the [original codebase]([link to original repo]). You should find out

- Where to find the pre-trained weights
- How to load the pre-trained weights into its corresponding model
- Trace one forward pass so that you know which classes and functions are required for a simple forward pass
  . Usually, you only have to reimplement those functions.
- Be able to locate the important components of the model: Where is the model class? Are there submodel 
  classes, *e.g.* EncoderModel, DecoderModel? Where is the self-attention layer? 
  Are there multiple different attention layers, *e.g.* *self-attention*, *cross-attention*...?
- How can you debug the model in the original environment of the repo? Do you have to set `print` statements
  or can you work with an interactive debugger like `ipdb`?
  
It is very important that before you start opening a PR in 🤗Transformers that you are able to **efficiently** 
debug code in the original repository! This means that you should be able to run a forward pass and print out the 
actual values of the output of a layer. *I.e* you are able to load a pre-trained model, pass an input vector of 
token ids, *i.e.* `input_ids = [0, 1, 4, 5, ...]` to the model's forward function and you are able to print out the 
intermediate outputs of - let's say - the first self-attention layer that could look something like this: 

```bash
[[[0.3427, 0.4756, ...], [-3.544, 0.3379, ...], ...], ...]
```

This means that your debugging environment should consists of a short script (ideally written by you) that 
does the following (in pseudocode):

```bash
model = [name of model]Model.load_pretrained_checkpoint(/path/to/checkpoint/)
input_ids = ... # vector of input ids
outputs = model.predict(input_ids)
```

By running such a script, you should be able to print out intermediate values or hit a break point
in the clone of the original repository that is saved locally on your computer.

We expect that every model addded to 🤗Transformers passes a couple of integration tests, meaning that the original 
model and the reimplemented version in 🤗Transformers have to give the exact same output up to a precision of 0.001! 
It is not enough if the model gives nearly the same output, they have to be the same. Therefore, you will 
certainly compare the intermediate outputs of the 🤗Transformers version multiple times against the intermediate outputs 
of the original implementation of [name of model] in which case an **effecient** debugging environment of the original 
repository is absolute key. Here is some advice is to make your debugging environment as efficient as possible.
	
- Find the best way of debugging intermediate results. Is the original repository written in PyTorch? Then you should be able 
  to use a simple debugger like [ipdb](https://pypi.org/project/ipdb/) to print out intermediate values. Is the original 
  repository written in Tensorflow 1? Then you might have to rely on tensorflow print operations like 
  https://www.tensorflow.org/api_docs/python/tf/print to output intermediate values. Is the original repository written 
  in Jax? Then make sure that the model is **not jitted** when running the forward pass, 
  *e.g.* check-out [this link](https://github.com/google/jax/issues/196).
- Use the smallest pre-trained checkpoint you can find. The smaller the checkpoint, the faster your debug cycle becomes. It is not efficient
  if your pre-trained model is so big that your forward pass takes more than 10 seconds. In case only very large checkpoints 
  are available, it might make more sense to create a dummy model in the new environment with randomly initialized weights 
  and save those weights for comparision with the 🤗Transformers version of your model
- Make sure you are using the easiest way of calling a forward pass in the original repository. Ideally, you want to find the function
  in the original repository that **only** calls a single forward pass, *i.e.* that is often called `predict`, `evaluate`, `forward` or `__call__`. 
  You don't want to debug a function that calls `forward` multiple times, *e.g.* to generate text, like `autoregressive_sample`, `generate`.
- Try to separate the tokenization from the model's `forward` pass. If the original repository shows examples where you have to input a string, then 
  try to find out where in the forward call the string input is changed to input ids and start from this point. This might mean that you have to possible 
  write a small script yourself or change the original code so that you can directly input the ids instead of an input string.
- Make sure that the model in your debugging setup is **not** in training mode, which often causes the model to yield random outputs due to 
  multiple dropout layers in the model. Make sure that the forward pass in your debugging environment is **deterministic** so that the dropout layers
  are not used.
	
The following section gives you more specific details/tips on how you can do this for [name of model].
	
### More details on how to create a debugging environment for [name of model] 
	
[Here you should add very specific information on what the student should do]
[to set up an efficient environment for the special requirements of this model]

## Implement [name of model] into 🤗Transformers
	
Next, you can finally add code to 🤗Transformers. Go into the clone 
of your 🤗Transformers' fork:

```
cd transformers
```

### Use the Cookiecutter to automatically generate the model's code

To begin with head over to 
the [🤗Transformers templates](https://github.com/huggingface/transformers/tree/master/templates/adding_a_new_model) to make
use of our `cookiecutter` implementation to automatically generate all the relevant 
files for your model. Again, we recommend to only add the PyTorch version of the model at first.
Make sure you follow the instructions of the `README.md` on the [🤗Transformers templates](https://github.com/huggingface/transformers/tree/master/templates/adding_a_new_model) carefully.

### Adapt the generated model's code for [name of model]
	
At first we will focus only on the model itself and not care about the tokenizer. All the relevant code should be found in 
the generated files `src/transformers/models/[lowercase name of model]/modeling_[lowercase name of model].py`
and `src/transformers/models/[lowercase name of model]/configuration_[lowercase name of model].py`.

Now you can finally start coding :). The generated code in `src/transformers/models/[lowercase name of model]/modeling_[lowercase name of model].py` will
either have the same architecture as BERT or BART if it's an encoder-decoder model.
At this point, you should remind yourself what you've learned in the beginning about the theoretical aspects of the model: *How is the model different from BERT or BART?*". Implement those changes which often means to change the *self-attention* layer, the order of the normalization layer, etc...
Here it is often useful to look at similar architecture of already existing models in Transformers.

**Note** that at this point, you don't have to be very sure that your code is fully correct or clean.
Rather, it is advised to add a first *unclean*, copy-pasted version of the original code to 
`src/transformers/models/[lowercase name of model]/modeling_[lowercase name of model].py` until you feel like all the 
necessary code is added. From our experience, it is much more efficient to quickly add a first version of the required code
and improve/correct the code iteratively with the conversion script as descriped in the next section. The only thing that has to work at this 
point is that you can instantiate the 🤗Transformers implementation of [name of model], *i.e.* the following command works:
	
```python 
from transformers import [camelcase name of model]Model, [camelcase name of model]Config

model = [camelcase name of model]Model([camelcase name of model]Config())
```

The above command will create a model according to the default parameters as defined in `[camelcase name of model]Config()`
with random weights, thus making sure that the `init()` methods of all components works.

In the case of [name of model], you should at least have to do the following changes:
	
[Here you should add very specific information on what exactly has to be changed for this model]
[...]
[...]

### Write a conversion script

Now, you should write a conversion script that let's you convert the checkpoint you used to debug 
[name of model] in the original repository to your just created 🤗Transformers implementation of [name of model].
Here you should not try to write the conversion script from scratch, but find similar models in 🤗Transformers
that require similar conversion scripts, *i.e.* whose original repository was written with the same framework as
[name of model].
	
In the conversion script, you should make sure that you set the parameters correctly in `[camelcase name of model]Config()`
to exactly match those that were used for the checkpoint you want to convert. Also it is very important that before
you set the retrieved weights to the new weight, *e.g.* via
```python
layer.weight.data = array
```
you make sure that both their **shape and name matches**. If either the shape or the name doesn't match, you probably assigned the wrong 
checkpoint weight to a randomely initialized layer of the 🤗Transformers implementation. Therefore, it is **absolutely necessary** to
add assert statements for the shape and print out the names of the checkpoints weights. E.g. you should add statements like:

```python
assert (
	pointer_random_weight.shape == checkpoint_weight.shape
), f"Pointer shape of random weight {pointer_random_weight.shape} and array shape of checkpoint weight {checkpoint_weight.shape} mismatched
```

and 

```python
logger.info(f"Initialize PyTorch weight {pointer_random_weight.name} from {checkpoint_weight.name}")
```

for verification. In addition, you should also check that **all** required weights are initialized and print
out all checkpoint weights that were not used for initialization to make sure the model is correctly converted.
It is completely normal, that the conversion trials fail with either a wrong shape statement or wrong name assignment.
This is most likely because either you used incorrect parameters in `[camelcase name of model]Config()`, have a wrong architecture 
in the the 🤗Transformers implementation, you have a bug in the `init()` functions of one the components of 
the 🤗Transformers implementation or you need to transpose one of the checkpoint weights.

This step should be iterated with the previous step until all weights of the chekpoint are correctly loaded in the Transformers model.

In the case of [name of model], you should probably do the following:
	
[Here you should add very specific information on what exactly has to be done for the conversion of this model]
[...]
[...]
	
### Implement the forward pass

TODO: PVP

### Implement the tokenizer

TODO: PVP

### Adding all necessary tests

TODO: PVP

### Refactor the added code

TODO: PVP

You have now finished the coding part, congratulation! 🎉 You are Awesome! 😎

## Open a Pull Request on the main huggingface/transformers repo

TODO: PVP

## Share your work!!

TODO: PVP


**You have made another model that is super easy to access for everyone in the community! 🤯**
