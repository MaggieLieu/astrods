"""galaxyMNIST dataset."""

import tensorflow_datasets as tfds
import h5py

_DESCRIPTION = """
High resolution GalaxyMNIST <https://github.com/mwalmsley/galaxy_mnist>`_ pytorch Dataset.

Based on MNIST/fashionMNIST datasets.

Galaxy images labelled by morphology (shape). Aimed at ML debugging and teaching.

Contains 10,000 images of galaxies (3x224x224), confidently labelled by Galaxy Zoo volunteers as belonging to one of four morphology classes.
"""

_CITATION = """
@ARTICLE{2022MNRAS.509.3966W,
       author = {{Walmsley}, Mike and {Lintott}, Chris and {G{\'e}ron}, Tobias and {Kruk}, Sandor and {Krawczyk}, Coleman and {Willett}, Kyle W. and {Bamford}, Steven and {Kelvin}, Lee S. and {Fortson}, Lucy and {Gal}, Yarin and {Keel}, William and {Masters}, Karen L. and {Mehta}, Vihang and {Simmons}, Brooke D. and {Smethurst}, Rebecca and {Smith}, Lewis and {Baeten}, Elisabeth M. and {Macmillan}, Christine},
        title = "{Galaxy Zoo DECaLS: Detailed visual morphology measurements from volunteers and deep learning for 314 000 galaxies}",
      journal = {\mnras},
     keywords = {methods: data analysis, galaxies: bar, galaxies: general, galaxies: interactions, Astrophysics - Astrophysics of Galaxies, Computer Science - Computer Vision and Pattern Recognition},
         year = 2022,
        month = jan,
       volume = {509},
       number = {3},
        pages = {3966-3988},
          doi = {10.1093/mnras/stab2093},
archivePrefix = {arXiv},
       eprint = {2102.08414},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.3966W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

Please also acknowledge the DECaLS survey
"""


class galaxyMNIST(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for galaxyMNIST dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(224, 224, 3)),
            'label': tfds.features.ClassLabel(names=['smooth & round', 'smooth & cigar-shaped', 'edge-on-disk', 'unbarred spiral']),
        }),
        supervised_keys=('image', 'label'),
        homepage='https://github.com/mwalmsley/galaxy_mnist',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path_train = dl_manager.download_and_extract('https://drive.google.com/uc?export=download&id=1MsK2lWOTJOk1P7rvXkOdFbyYXkuSDYcf')
    path_test = dl_manager.download_and_extract('https://drive.google.com/uc?export=download&id=1YxyH58ClJ52y8-KGQZWcV6sCDnHsCx8B')

    return {
        'train': self._generate_examples(path_train),
        'test': self._generate_examples(path_test),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    with h5py.File(path,'r') as f:
        images = f['images'][:]
        targets = f['labels'][:]
        for k, (i,j) in enumerate(zip(images,targets)):
            yield k, {
                'image': i,
                'label': j,
            }
