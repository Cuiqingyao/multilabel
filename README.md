# HMIO: A Hierarchical multi-input and output Bi-GRU Model

we propose a hierarchical multi-input and output model based bi-directional recurrent neural network, which both considers the semantic and lexical information of emotional expression. Our model applies two independent Bi-GRU layer to generate part of speech and sentence representation. Then the lexical information is considered via attention over output of softmax activation on part of speech representation.


### Requirements
- `gensim`
- `numpy`
- `tensorflow==1.1.0`
- `Keras==2.0.4`
- `scikit-learn`
- `h5py`
- `argparse`
- `PyYAML`

### How to use

- **install virtualenv**
```bash
$ pip install virtualenv

# cd project
$ cd /your/path/to/project/HMIO

# make env using python 3.5
$ virtualenv env --python=python3

# attention: if you want to use gpu,env's name must be 'tensorflow',maybe like this:
$ virtualenv tensorflow --python=python

# you will see a new dir named “env”; activate the env by followed cmd:
$ source env/bin/activate
```

- **The requirements can be installed via `pip` as follows:**

```bash
$ pip install -r requirements.txt
```
