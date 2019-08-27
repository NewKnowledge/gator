import os
import sys
import typing
import numpy as np
import pandas as pd
import time
from d3m.primitive_interfaces.base import CallResult, PrimitiveBase
from nk_imagenet import ImagenetModel

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp
from sklearn.preprocessing import LabelEncoder

__author__ = 'Distil'
__version__ = '1.0.1'
__contact__ = 'mailto:nklabs@newknowledge.io'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    batch_size = hyperparams.UniformInt(
        lower = 1, 
        upper = 256,
        upper_inclusive=True, 
        default = 16, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'batch size'
    )
    top_layer_epochs = hyperparams.UniformInt(
        lower = 1, 
        upper = sys.maxsize,
        default = 10, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of epochs to finetune top layer'
    )
    all_layer_epochs = hyperparams.UniformInt(
        lower = 1, 
        upper = sys.maxsize,
        default = 50, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of epochs to finetune top m - n layers, where m is total number of layers'
    )
    frozen_layer_count = hyperparams.UniformInt(
        lower = 1, 
        upper = sys.maxsize,
        default = 249, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of layers, n, to keep frozen'
    )
    pooling = hyperparams.Enumeration(
        default = 'avg', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        values = ['avg', 'max'],
        description = 'whether to use average or max pooling to transform 4D ImageNet features to 2D output'
    )
    include_class_weights = hyperparams.UniformBool(
        default = True, 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="whether to include class weights in finetuning of ImageNet model"
    )


class gator(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Produce image classification predictions by clustering an Inception model
        finetuned on all columns of images in the dataset (assumption = single column of target labels)

        Parameters
        ----------
        inputs : d3m dataframe with columns of image paths and optional labels

        Returns
        -------
        output : A dataframe with image labels/classifications/cluster assignments
    """

    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "475c26dc-eb2e-43d3-acdb-159b80d9f099",
        'version': __version__,
        'name': "gator",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Image Recognition', 'digital image processing', 'ImageNet', 'Convolutional Neural Network'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/gator",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        "installation": [
            {
                "type": "PIP",
                "package_uri": "git+https://github.com/NewKnowledge/gator.git@{git_commit}#egg=Gator".format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__))
                ),
            },
            {
            "type": "FILE",
            "key": "gator_weights",
            "file_uri": "http://public.datadrivendiscovery.org/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
            "file_digest":"9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a"
            },
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.digital_image_processing.imagenet_convolutional_neural_network.Gator',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        "algorithm_types": [
            metadata_base.PrimitiveAlgorithmType.IMAGENET_CONVOLUTIONAL_NEURAL_NETWORK
        ],
        "primitive_family": metadata_base.PrimitiveFamily.DIGITAL_IMAGE_PROCESSING
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str,str]=None)-> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)
        self.ImageNet = ImagenetModel(weights = self.volumes["gator_weights"], pooling = self.hyperparams['pooling'])
        self.image_paths = None
        self.image_labels = None
        self.class_weights = None
        self.targets = None

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
            Sets primitive's training data
            
            Parameters
            ----------
            inputs: column(s) of image paths
            outputs: labels from dataframe's target column
        '''
        
        # create single list of image paths from all target image columns
        image_cols =inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        base_paths = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t))['location_base_uris'][0].replace('file:///', '/') for t in image_cols]
        self.image_paths = np.array([[os.path.join(base_path, filename) for filename in inputs.iloc[:,col]] for base_path, col in zip(base_paths, image_cols)]).flatten()

        # broadcast image labels for each column of images
        self.targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(self.targets):
            self.targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Target')
        if not len(self.targets):
            self.targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')

        # assert that the number of image columns is less than or equal to the number of target columns in the df
        assert len(image_cols) <= len(self.targets), "List of image columns cannot be longer than list of target columns"

        # train label encoder
        self.encoder = LabelEncoder().fit(inputs.iloc[:,self.targets[0]])
        self.image_labels = self.encoder.transform(np.repeat(inputs.iloc[:,self.targets[0]], len(image_cols)))

        # calculate class weights for target labels if desired
        if self.hyperparams['include_class_weights']:
           self.class_weights = dict(pd.Series(self.image_labels).value_counts())

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
            Trains a single Inception model on all columns of image paths using dataframe's target column
        '''
        start_time = time.time()
        print('finetuning begins!', file = sys.__stdout__)
        self.ImageNet.finetune(
            self.image_paths, 
            self.image_labels,
            nclasses = len(self.encoder.classes_),
            batch_size = self.hyperparams['batch_size'],
            pooling = self.hyperparams['pooling'],
            top_layer_epochs = self.hyperparams['top_layer_epochs'],
            all_layer_epochs = self.hyperparams['all_layer_epochs'],
            frozen_layer_count = self.hyperparams['frozen_layer_count'],
            class_weight = self.class_weights)
        print(f'finetuning ends!. it took {time.time()-start_time} seconds', file = sys.__stdout__)
        return CallResult(None)        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
            Produce image object classification predictions

            Parameters
            ----------
            inputs : d3m dataframe with columns of image paths and optional labels

            Returns
            -------
            output : A dataframe with image labels/classifications/cluster assignments
        """
    
        # get metadata labels for primary key and target label columns
        key = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        col_names = [inputs.metadata.query_column(key[0])['name']]
        target_names = [inputs.metadata.query_column(idx)['name'] for idx in self.targets]

        # create output dataframe
        result_df = d3m_DataFrame(pd.DataFrame(columns=col_names.extend(target_names)))
        result_df[col_names[0]] = inputs[col_names[0]]
        col_dict = dict(result_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type(1)
        col_dict['name'] = col_names[0]
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        result_df.metadata = result_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)

        image_cols =inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        for idx, col in enumerate(image_cols):
            base_path = inputs.metadata.query((metadata_base.ALL_ELEMENTS, col))['location_base_uris'][0].replace('file:///', '/')
            image_paths = np.array([os.path.join(base_path, filename) for filename in inputs.iloc[:,col]])

            # make predictions on finetuned model and decode
            preds = self.ImageNet.finetuned_predict(image_paths)
            result_df[target_names[idx]] = self.encoder.inverse_transform(np.argmax(preds, axis=1))

            # add metadata to column
            col_dict = dict(result_df.metadata.query((metadata_base.ALL_ELEMENTS, idx+1)))
            col_dict['structural_type'] = type(1)
            col_dict['name'] = target_names[idx]
            col_dict['semantic_types'] = ('http://schema.org/Integer', 
                                        'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 
                                        'https://metadata.datadrivendiscovery.org/types/TrueTarget', 
                                        'https://metadata.datadrivendiscovery.org/types/Target')
            result_df.metadata = result_df.metadata.update((metadata_base.ALL_ELEMENTS, idx+1), col_dict)
        
        return CallResult(result_df)

    
