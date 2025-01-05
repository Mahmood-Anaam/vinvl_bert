from setuptools import setup, find_packages

setup(
    name="vinvl_bert",
    version="0.1.0",
    description="Arabic Image Captioning using Pre-training of Deep Bidirectional Transformers",
    author="Mahmood Anaam",
    author_email="eng.mahmood.anaam@gmail.com",
    url="https://github.com/Mahmood-Anaam/vinvl_bert",
    license="MIT",
    packages=find_packages(exclude=["notebooks", "assets", "scripts", "tests"]),
    install_requires = [
        "vinvl @ git+https://github.com/Mahmood-Anaam/vinvl.git",
        "anytree==2.12.1",
        "cityscapesScripts==2.2.4",
        "clint==0.5.1",
        "Cython==3.0.11",
        "einops==0.8.0",
        "huggingface-hub==0.27.0",
        "ninja==1.11.1.3",
        "opencv-python==4.10.0.84",
        "pillow==11.1.0",
        "PyYAML==6.0.2",
        "timm==1.0.12",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "tqdm==4.67.1",
        "transformers==4.47.1",
        "pytorch-transformers==1.2.0",
        "twine==6.0.1",
        "urllib3==2.3.0",
        "yacs==0.1.8",
        ],


    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)

