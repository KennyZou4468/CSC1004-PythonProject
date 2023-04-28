import torch.multiprocessing as mp
import torch
import statistics
import matplotlib as plt
def test(a,b):
    print(a)
    print(b)
    print(a+b)
if __name__ == '__main__':
    print(torch.cuda.is_available())
    c=[1,2,3,4,5,6]
    print(statistics.mean(c))


