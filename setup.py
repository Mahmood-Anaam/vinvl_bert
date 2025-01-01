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
    install_requires=[
        "yacs",
        "torch",
        "torchvision",
        "transformers",
        "pytorch-transformers",
        "numpy",
        "opencv-python",
        "Pillow",
        "tqdm",
        "anytree",
        "pycocotools",
        "timm",
        "einops",
        "PyYAML",
        "cython",
        "ninja",
        "clint",
        "cityscapesScripts",
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)

