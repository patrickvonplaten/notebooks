# How to add [<name of model>] to 🤗Transformers
	
Teacher: <name of hugging face teacher>
	
Begin: <start date>
	
Estimated time: <end date>

The following sections explain in detail how to add <name of model> 
to 🤗Transformers. You will work closely with <name of hugging face teacher> to
integrate <name of model> into 🤗Transformers. By doing so, you will both gain a 
theoratical and deep practical understanding of <name of model>. But more importantly, 
you will have made a major open-source contribution to 🤗Transformers. Along the way,
you will:
	
- get insights into open-source best practices, 
- understand the design principles of one of the most popular NLP libraries,
- learn how to do efficiently test large NLP models,
- learn how to integrate python utilities like `black`, `isort`, `make fix-copies` into a library 
  to always ensure clean and readable code.

To begin with, you should start by getter a good understanding of the model.
  
## Theoritacl aspects of <name of model>
  
### Paper

  You should take some time to read [<name of model>'s paper](<link to paper>).
  There might be large sections of the paper that are difficult to understand.
  If this is the case, this is totally fine - don't worry! The goal is not to get 
  a deep theoretical understanding of the paper, but to extract the necessary information 
  required to effectively reimplement the model to 🤗Transformers.
  That being said, you don't have to spend too much time on the theoretical aspects,
  but rather focus on the practical ones, namely:
  
  - What time of model is <name of model>? BERT-like encoder-only model? GPT2-like decoder-only model? BART-like encoder-decoder model?
  - What are the applications of <name of model>? Text classification? Text generation? Seq2Seq tasks, *e.g.* summarization?
  - What is the novel feature of the model making it different to BERT, RoBERTa, or BART?
  - Which of the already existing [🤗Transformers models](https://huggingface.co/transformers/#contents) is most similar to <name of model>?
  
 After you feel like you have gotten a good overview over the architecture of the model, you might want 
 to ping <name of hugging face teacher> for any questions you might have.
 This might include questions regarding the model's architecture, its attention layer, etc. <name of hugging face teacher> will be more 
 than happy to help you.
 
### Additional resources

 Before diving into the code, here are some additional resources that might be worth taking a look at:
 
 - <link 1>
 - <link 2>
 - <link 3>
 - ...
 
### Make sure you've understood the fundamental aspects of <name of model>
 
Alright, now you should be ready to take a closer look into the actualy code of <name of model>.
You should have understood the following aspects of <name of model> by now:

- <characteristic 1 of name of model>
- <characteristic 2 of name of model>
- ...

If any of the mentioned aspects above are **not** clear to you, now is a great time to talk to <name of hugging face teacher> again!

## Next prepare your environment

1. Fork the [repository](https://github.com/huggingface/transformers) by clicking on the 'Fork' button on the repository's page.
This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

	```bash
	git clone https://github.com/<your Github handle>/transformers.git
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

4. We recommend to add the PyTorch version of <name of model> to Transformers. In order to install PyTorch,
  please follow the instructions on: https://pytorch.org/. 
  
  **Note:** You don't need to have CUDA install. It is sufficient to just be working on CPU.

5. To port <name of model>, you will also need access to its [original repository](<link to original repo>):
  
  ```bash
  git clone <clone link to original repo>
  cd <name of repo>
  pip install -e .
  ```

Now you have set up a development environment to port <name of model> to 🤗Transformers.
  
## Run a pre-trained checkpoint using the original repository

At first, you will work the original repository. Often, the original implementation is very
 "researchy" meaning that documentation might be lacking and the code can be hard to read / understand.
But this should be exactly your motivation to reimplement <name of the model>. At Hugging Face, one of our main goals is to
*make people stand on the shoulders of giants* which translates here very well into taking a working 
model and rewriting it to make it as **accesable, user-friendly, and beautiful** as possible.
This is the #1 motivation to reimplement models into 🤗Transformers - trying to maximize access 
to a complex new NLP technology for **everybody**.
	
You should start thereby by diving into the original repository.

### Get familiar with the original repository.

Succesfully running the official pre-trained model in the original repository is often 
**the most difficult** step. From our experience, it is very important to spend some time to 
get familiar with the [original codebase](<link to original repo>). You should find out

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
model = <name of model>Model.load_pretrained_checkpoint(/path/to/checkpoint/)
input_ids = ... # vector of input ids
outputs = model.predict(input_ids)
```

By running such a script, you should be able to print out intermediate values or hit a break point
in the clone of the original repository that is saved locally on your computer.

We expect that every model addded to 🤗Transformers passes a couple of integration tests, meaning that the original 
model and the reimplemented version in 🤗Transformers have to give the exact same output up to a precision of 0.001! 
It is not enough if the model gives nearly the same output, they have to be the same. Therefore, you will 
certainly compare the intermediate outputs of the 🤗Transformers version multiple times against the intermediate outputs 
of the original implementation of <name of model> in which case an **effecient** debugging environment of the original 
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
	
The following section gives you more specific details/tips on how you can do this for <name of model>.
	
### More details on how to create a debugging environment for <name of model> 
	
<Here you should add very specific information on what the student should do>
<to set up an efficient environment for the special requirements of this model>

## Implement <name of model> into 🤗Transformers
	
Next, you should write the code 


## Adding a new dataset

### Understand the structure of the dataset

1. Find a short-name for the dataset:

	- Select a `short name` for the dataset which is unique but not too long and is easy to guess for users, e.g. `squad`, `natural_questions`
	- Sometimes the short-list name is already given/proposed (e.g. in the spreadsheet of the data sprint to reach v2.0 if you are participating in the effort)

You are now ready to start the process of adding the dataset. We will create the following files:

- a **dataset script** which contains the code to download and pre-process the dataset: e.g. `squad.py`,
- a **dataset card** with tags and information on the dataset in a `README.md`.
- a **metadata file** (automatically created) which contains checksums and informations about the dataset to guarantee that the loading went fine: `dataset_infos.json`
- a **dummy-data file** (automatically created) which contains small examples from the original files to test and garantee that the script is working well in the future: `dummy_data.zip`

2. Let's start by creating a new branch to hold your development changes with the name of your dataset:

	```bash
	git fetch upstream
	git rebase upstream/master
	git checkout -b a-descriptive-name-for-my-changes
	```

	**Do not** work on the `master` branch.

3. Create your dataset folder under `datasets/<your_dataset_name>`:

	```bash
	mkdir ./datasets/<your_dataset_name>
	```

4. Open a new online [dataset card form](https://huggingface.co/datasets/card-creator/) to fill out: you will be able to download it to your dataset folder with the `Export` button when you are done. Alternatively, you can also manually create and edit a dataset card in the folder by copying the template:

	```bash
	cp ./templates/README.md ./datasets/<your_dataset_name>/README.md
	```

5. Now explore the dataset you have selected while completing some fields of the **dataset card** while you are doing it:

	- Find the research paper or description presenting the dataset you want to add
	- Read the relevant part of the paper/description presenting the dataset
	- Find the location of the data for your dataset
	- Download/open the data to see how it looks like
	- While you explore and read about the dataset, you can complete some sections of the dataset card (the online form or the one you have just created at `./datasets/<your_dataset_name>/README.md`). You can just copy the information you meet in your readings in the relevant sections of the dataset card (typically in `Dataset Description`, `Dataset Structure` and `Dataset Creation`).

		If you need more information on a section of the dataset card, a detailed guide is in the `README_guide.md` here: https://github.com/huggingface/datasets/blob/master/templates/README_guide.md.

		There is a also a (very detailed) example here: https://github.com/huggingface/datasets/tree/master/datasets/eli5.

		Don't spend too much time completing the dataset card, just copy what you find when exploring the dataset documentation. If you can't find all the information it's ok. You can always spend more time completing the dataset card while we are reviewing your PR (see below) and the dataset card will be open for everybody to complete them afterwards. If you don't know what to write in a section, just leave the `[More Information Needed]` text.


### Write the loading/processing code

Now let's get coding :-)

The dataset script is the main entry point to load and process the data. It is a python script under `datasets/<your_dataset_name>/<your_dataset_name>.py`.

There is a detailed explanation on how the library and scripts are organized [here](https://huggingface.co/docs/datasets/add_dataset.html).

Note on naming: the dataset class should be camel case, while the dataset short_name is its snake case equivalent (ex: `class BookCorpus` for the dataset `book_corpus`).

To add a new dataset, you can start from the empty template which is [in the `templates` folder](https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py):

```bash
cp ./templates/new_dataset_script.py ./datasets/<your_dataset_name>/<your_dataset_name>.py
```

And then go progressively through all the `TODO` in the template 🙂. If it's your first dataset addition and you are a bit lost among the information to fill in, you can take some time to read the [detailed explanation here](https://huggingface.co/docs/datasets/add_dataset.html).

You can also start (or copy any part) from one of the datasets of reference listed below. The main criteria for choosing among these reference dataset is the format of the data files (JSON/JSONL/CSV/TSV/text) and whether you need or don't need several configurations (see above explanations on configurations). Feel free to reuse any parts of the following examples and adapt them to your case:

- question-answering: [squad](https://github.com/huggingface/datasets/blob/master/datasets/squad/squad.py) (original data are in json)
- natural language inference: [snli](https://github.com/huggingface/datasets/blob/master/datasets/snli/snli.py) (original data are in text files with tab separated columns)
- POS/NER: [conll2003](https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py) (original data are in text files with one token per line)
- sentiment analysis: [allocine](https://github.com/huggingface/datasets/blob/master/datasets/allocine/allocine.py) (original data are in jsonl files)
- text classification: [ag_news](https://github.com/huggingface/datasets/blob/master/datasets/ag_news/ag_news.py) (original data are in csv files)
- translation: [flores](https://github.com/huggingface/datasets/blob/master/datasets/flores/flores.py) (original data come from text files - one per language)
- summarization: [billsum](https://github.com/huggingface/datasets/blob/master/datasets/billsum/billsum.py) (original data are in json files)
- benchmark: [glue](https://github.com/huggingface/datasets/blob/master/datasets/glue/glue.py) (original data are various formats)
- multilingual: [xquad](https://github.com/huggingface/datasets/blob/master/datasets/xquad/xquad.py) (original data are in json)
- multitask: [matinf](https://github.com/huggingface/datasets/blob/master/datasets/matinf/matinf.py) (original data need to be downloaded by the user because it requires authentificaition)

While you are developping the dataset script you can list test it by opening a python interpreter and running the script (the script is dynamically updated each time you modify it):

```python
from datasets import load_dataset

data = load_dataset('./datasets/<your_dataset_name>')
```

This let you for instance use `print()` statements inside the script as well as seeing directly errors and the final dataset format.

**What are configurations and splits**

Sometimes you need to use several *configurations* and/or *splits* (usually at least splits will be defined).

* Using several **configurations** allow to have like sub-datasets inside a dataset and are needed in two main cases:

	- The dataset covers or group several sub-datasets or domains that the users may want to access independantly and/or
	- The dataset comprise several sub-part with different features/organizations of the data (e.g. two types of CSV files with different types of columns). Inside a configuration of a dataset, all the data should have the same format (columns) but the columns can change accross configurations.

* **Splits** are a more fine grained division than configurations. They allow you, inside a configuration of the dataset, to split the data in typically train/validation/test splits. All the splits inside a configuration should have the same columns/features and splits are thus defined for each specific configurations of there are several.


**Some rules to follow when adding the dataset**:

- try to give access to all the data, columns, features and information in the dataset. If the dataset contains various sub-parts with differing formats, create several configurations to give access to all of them.
- datasets in the `datasets` library are typed. Take some time to carefully think about the `features` (see an introduction [here](https://huggingface.co/docs/datasets/exploring.html#features-and-columns) and the full list of possible features [here](https://huggingface.co/docs/datasets/features.html))
- if some of you dataset features are in a fixed set of classes (e.g. labels), you should use a `ClassLabel` feature.


**Last step:** To check that your dataset works correctly and to create its `dataset_infos.json` file run the command:

```bash
python datasets-cli test datasets/<your-dataset-folder> --save_infos --all_configs
```

**Note:** If your dataset requires manually downloading the data and having the user provide the path to the dataset you can run the following command:
```bash
python datasets-cli test datasets/<your-dataset-folder> --save_infos --all_configs --data_dir your/manual/dir
```
To have the configs use the path from `--data_dir` when generating them.

### Automatically add code metadata

Now that your dataset script runs and create a dataset with the format you expected, you can add the JSON metadata and test data.

**Make sure you run all of the following commands from the root of your `datasets` git clone.**

1. To create the dummy data for continuous testing, there is a tool that automatically generates dummy data for you. At the moment it supports data files in the following format: txt, csv, tsv, jsonl, json, xml.
If the extensions of the raw data files of your dataset are in this list, then you can automatically generate your dummy data with:

	```bash
 	python datasets-cli dummy_data datasets/<your-dataset-folder> --auto_generate
	```

	Example:

	```bash
 	python datasets-cli dummy_data ./datasets/snli --auto_generate
	```

	If this doesn't work more information on how to add dummy data can be found in the documentation [here](https://huggingface.co/docs/datasets/share_dataset.html#adding-dummy-data).

If you've been fighting with dummy data creation without success for some time and can't seems to make it work:
Go to the next step (open a Pull Request) and we'll help you cross the finish line 🙂

2. Now test that both the real data and the dummy data work correctly using the following commands:

	*For the real data*:
	```bash
	RUN_SLOW=1 pytest tests/test_dataset_common.py::LocalDatasetTest::test_load_real_dataset_<your-dataset-name>
	```
	and

	*For the dummy data*:
	```bash
	RUN_SLOW=1 pytest tests/test_dataset_common.py::LocalDatasetTest::test_load_dataset_all_configs_<your-dataset-name>
	```

	On **Windows**, you may need to run:
	```
	$Env:RUN_SLOW = "1"
	pytest tests/test_dataset_common.py::LocalDatasetTest::test_load_real_dataset_<your-dataset-name>
	pytest tests/test_dataset_common.py::LocalDatasetTest::test_load_dataset_all_configs_<your-dataset-name>
	```
	to enable the slow tests, instead of `RUN_SLOW=1`.

3. If all tests pass, your dataset works correctly. You can finally create the metadata JSON by running the command:

	```bash
	python datasets-cli test datasets/<your-dataset-folder> --save_infos --all_configs
	```

	This first command should create a `dataset_infos.json` file in your dataset folder.


You have now finished the coding part, congratulation! 🎉 You are Awesome! 😎

### Open a Pull Request on the main HuggingFace repo and share your work!!

Here are the step to open the Pull-Request on the main repo.

1. Format your code. Run black, isort and flake8 so that your newly added files look nice with the following commands:

	```bash
	make style
	flake8 datasets
	```

	If you are on windows and `make style` doesn't work you can do the following steps instead:

	```bash
	pip install black
	pip install isort
	pip install flake8

	black --line-length 119 --target-version py36 datasets/your_dataset
	
	isort datasets/your_dataset/your_dataset.py

	flake8 datasets/your_dataset
	```

2. Make sure that you have a dataset card (more information in the [next section](#tag-the-dataset-and-write-the-dataset-card)) with:

	1. **Required:** The YAML tags obtained with the [tagging app](https://github.com/huggingface/datasets-tagging) and a description of the various fields in your dataset.
	2. Any relevant information you would like to share with users of your dataset in the appropriate paragraphs.

3. Once you're happy with your dataset script file, add your changes and make a commit to record your changes locally:

	```bash
	git add datasets/<your_dataset_name>
	git commit
	```

	It is a good idea to sync your copy of the code with the original
	repository regularly. This way you can quickly account for changes:
	
	- If you haven't pushed your branch yet, you can rebase on upstream/master:

	  ```bash
	  git fetch upstream
	  git rebase upstream/master
	  ```
	  
	- If you have already pushed your branch, do not rebase but merge instead:
	
	  ```bash
	  git fetch upstream
	  git merge upstream/master
	  ```

   Push the changes to your account using:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

3. Once you are satisfied, go the webpage of your fork on GitHub. Click on "Pull request" to send your to the project maintainers for review.

Congratulation you have open a PR to add a new dataset 🙏

**Important note:** In order to merge your Pull Request the maintainers will require you to tag and add a dataset card. Here is now how to do this last step:

### Tag the dataset and write the dataset card

Each dataset is provided with a dataset card.

The dataset card and in particular the tags which are on it are **really important** to make sure the dataset can be found on the hub and will be used by the users. Users need to have the best possible idea of what's inside the dataset and how it was created so that they can use it safely and have a good idea of the content.

Creating the dataset card goes in two steps:

1. **Tagging the dataset using the tagging streamlit app**

	Clone locally the dataset-tagging app which is here: https://github.com/huggingface/datasets-tagging

	Run the app with the command detailed in the readme: https://github.com/huggingface/datasets-tagging/blob/main/README.md

	Enter the full path to your dataset folder on the left, and tag the different configs :-) (And don't forget to save to file after you're done with a config!)

2. **Copy the tags in the dataset card and complete the dataset card**

	- **Essential:** Once you have saved the tags for all configs, you can expand the **Show YAML output aggregating the tags** section on the right, which will show you a YAML formatted block to put in the relevant section of the [online form](https://huggingface.co/datasets/card-creator/) (or manually  paste into your README.md).

	- **Very important as well:** On the right side of the tagging app, you will also find an expandable section called **Show Markdown Data Fields**. This gives you a starting point for the description of the fields in your dataset: you should paste it into the **Data Fields** section of the [online form](https://huggingface.co/datasets/card-creator/) (or your local README.md), then modify the description as needed. Briefly describe each of the fields and indicate if they have a default value (e.g. when there is no label). If the data has span indices, describe their attributes (character level or word level, contiguous or not, etc). If the datasets contains example IDs, state whether they have an inherent meaning, such as a mapping to other datasets or pointing to relationships between data points.

		Example from the [ELI5 card](https://github.com/huggingface/datasets/tree/master/datasets/eli5#data-fields):

			Data Fields:
				- q_id: a string question identifier for each example, corresponding to its ID in the Pushshift.io Reddit submission dumps.
				- subreddit: One of explainlikeimfive, askscience, or AskHistorians, indicating which subreddit the question came from
				- title: title of the question, with URLs extracted and replaced by URL_n tokens
				- title_urls: list of the extracted URLs, the nth element of the list was replaced by URL_n


	- **Very nice to have but optional for now:** Complete all you can find in the dataset card using the detailed instructions for completed it which are in the `README_guide.md` here: https://github.com/huggingface/datasets/blob/master/templates/README_guide.md.

		Here is a completed example: https://github.com/huggingface/datasets/tree/master/datasets/eli5 for inspiration

		If you don't know what to write in a field and can find it, write: `[More Information Needed]`

If you are using the online form, you can then click the `Export` button at the top to download a `README.md` file to your data folder. Once your `README.md` is ok you have finished all the steps to add your dataset, congratulation your Pull Request can be merged.

**You have made another dataset super easy to access for everyone in the community! 🤯**
