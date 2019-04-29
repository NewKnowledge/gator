from setuptools import setup, find_packages

setup(
    name="gator",
    version="1.0.0",
    description="Generates Automatic Tags via Object Recognition",
    author="New Knowledge",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["pandas", "pytest"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/imagenet.git@24e605be4f32336ab888f48ae37da95a3dd5d1df#egg=nk_imagenet",
    ],
)
