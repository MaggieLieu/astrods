"""XCLASS dataset."""

import tensorflow_datasets as tfds
import csv

_DESCRIPTION = """
The XMM CLuster Archive Super Survey is an X-ray galaxy cluster search in the XMM-Newton Science Archive (ESA). The search is applied to XMM-Newton X-ray 10-20ks observations. X-Amin pipeline is applied to spatially filter out sources of interest and with the addition of optical imaging from DSS2, manual selection is used to classify clusters ( high and low redshift) from other sources (galaxies, AGN, point sources, artefacts etc.). Here we provide the X-ray and optical images of the pipeline extractions and the class label. This dataset was used in the Hunt for clusters zooniverse project. Please cite Kosiba+2020 if you use this dataset.

The image data contains 3 channels [empty, Optical, X-ray]
labels correspond to: ['low-z', 'hi-z', 'galaxy', 'point', 'other']
"""

_CITATION = """
@article{kosiba2020multiwavelength,
  title={Multiwavelength classification of X-ray selected galaxy cluster candidates using convolutional neural networks},
  author={Kosiba, Matej and Lieu, Maggie and Altieri, Bruno and Clerc, Nicolas and Faccioli, Lorenzo and Kendrew, Sarah and Valtchanov, Ivan and Sadibekova, Tatyana and Pierre, Marguerite and Hroch, Filip and others},
  journal={Monthly Notices of the Royal Astronomical Society},
  volume={496},
  number={4},
  pages={4141--4153},
  year={2020},
  publisher={Oxford University Press}
}
"""

class XCLASS(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for XCLASS dataset."""

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
            'image': tfds.features.Image(shape=(356, 356, 3)),
            'label': tfds.features.ClassLabel(names=['low-z', 'hi-z', 'galaxy', 'point', 'other']),
        }),
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://xmm-xclass.in2p3.fr/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract('https://zenodo.org/record/7236582/files/Kosiba2020.zip?download=1')
    
    return {
        'train': self._generate_examples(
            images_path = path / 'Kosiba2020/train_images',
            label_path = path / 'Kosiba2020/train.csv'),
        'test': self._generate_examples(
            images_path = path / 'Kosiba2020/test_images',
            label_path = path / 'Kosiba2020/test.csv'),
    }
    


  def _generate_examples(self, images_path, label_path):
    """Yields examples."""
    with label_path.open() as f:
        for row in csv.DictReader(f):
          image_id = row['image_id']
          yield image_id, {
              'image': images_path / f'{image_id}.png',
              'label': row['class'],
          }
