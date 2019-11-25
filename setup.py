from distutils.core import setup


setup(
    name='gator',
    version='1.0.1',
    description='Generates Automatic Tags via Object Recognition',
    packages=['gator'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas>=0.23.4,<=0.25.2',    
        'tensorflow-gpu == 2.0.0',
        'nk_imagenet @ git+https://github.com/NewKnowledge/imagenet.git@9f6d6aaf9e115f346cda9255b5920cf2f39ff717#egg=nk_imagenet'
    ],
    entry_points={
        'd3m.primitives': [
            'digital_image_processing.convolutional_neural_net.Gator = gator:gator'
        ],
    }
)
