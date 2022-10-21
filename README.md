# astrods
astrods provides a collection of ready-to-use datasets in Astronomy for use with TensorFlow, Jax, and other Machine Learning frameworks. The datasets have been built with the tensorflow-datasets (tfds) dataset builder and therefore it handles the downloading and preparing of the data deterministically and the construction of the tf.data.Dataset object. 

# data availability
The current datasets available are:

| Name					| reference				| size 		| splits 					|
| --------------------- | --------------------- | --------- | -------------------------	|
| XCLASS 				| Kosiba et al. (2020)  | 259.3 MB	| ('train', 'test')			|



# requirements

astrods requires tensorflow and tensorflow-datasets to be installed `pip install tensorflow-datasets`

# Usage
Import libraries and download data
```
import astrods.XCLASS #replace XCLASS with dataset of interest
import tensorflow_datasets as tfds

ds, info = tfds.load('XCLASS', split='train', shuffle_files=True, with_info=True) #replace XCLASS with dataset of interest
print(info)
```

Example datasample
```
sample = ds.take(1)
for ex in sample:
	print(ex)
```
