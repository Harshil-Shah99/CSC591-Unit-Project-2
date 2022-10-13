# Script for CSC591 Unit Project 2 using the MNIST dataset

The initial model for this project is taken from:  https://github.com/pytorch/examples/tree/main/mnist upon which I add code for the quantization.

In this repository, you will find the requirements.txt file. Please install the requirements from that file as shown below:

```bash
pip install -r requirements.txt
python main.py
```

The file main.py contains all of the code required. Part of the code is from the github repo showing the example model training, and the other part is added by me to experiment with different types of quantization and different configurations.

The results, summarized, are in four jpeg images named base.jpg, naive.jpg, qat.jpg, and dorefa.jpg
