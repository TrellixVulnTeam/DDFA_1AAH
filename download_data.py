from datasets import CUB200, StanfordDogs,MNIST

# CUB200(root='./data/cub200', split='train',
#                     download=True)
# StanfordDogs(root='./data/dogs', split='train',
#                     download=True)
MNIST(root='./data',split='train')