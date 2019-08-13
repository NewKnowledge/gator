from distutils.core import setup


setup(
    name='gator',
    version='1.0.1',
    description='Generates Automatic Tags via Object Recognition',
    packages=['Gator'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas == 0.23.4',        
        'nk_imagenet @ git+https://github.com/NewKnowledge/imagenet.git@262fea27819a8e76e7e278dfc0f91f8f3b4ff392#egg=nk_imagenet'
    ],
    entry_points={
        'd3m.primitives': [
            'digital_image_processing.imagenet_convolutional_neural_network.Gator = object_recognition:gator'
        ],
    }
)