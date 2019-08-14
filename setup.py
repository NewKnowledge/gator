from distutils.core import setup


setup(
    name='gator',
    version='1.0.1',
    description='Generates Automatic Tags via Object Recognition',
    packages=['gator'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas == 0.23.4',        
        'nk_imagenet @ git+https://github.com/NewKnowledge/imagenet.git@7775abf69ba65bcf434f02eea41780ee146fa74f#egg=nk_imagenet'
    ],
    entry_points={
        'd3m.primitives': [
            'digital_image_processing.imagenet_convolutional_neural_network.Gator = object_recognition:gator'
        ],
    }
)