from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
from nni.algorithms.compression.pytorch.quantization import NaiveQuantizer
from nni.algorithms.compression.pytorch.quantization import DoReFaQuantizer
import time
import copy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    start = time.time()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    end = time.time()
    print('Time taken to test: {:.4f}'.format(end - start))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    copied_model_base = copy.deepcopy(model)
    copied_model_naive_quantizer_1 = copy.deepcopy(model)
    copied_model_naive_quantizer_2 = copy.deepcopy(model)
    copied_model_qat_quantizer_1 = copy.deepcopy(model)
    copied_model_qat_quantizer_2 = copy.deepcopy(model)
    copied_model_dorefa_quantizer_1 = copy.deepcopy(model)
    copied_model_dorefa_quantizer_2 = copy.deepcopy(model)

    print('----------------------------------------------------------------------')
    print('Experiment Results:')
    print('----------------------------------------------------------------------')
    print('\nBase Model without Quantization:')
    print('Model print output:')
    print(copied_model_base)
    test(copied_model_base, device, test_loader)

    print('----------------------------------------------------------------------')
    print('Naive Quantizer 1 with 8 bit weight, linear operation type:')
    print('Model print output:')
    config_list_naive_quantizer_1 = [{'quant_types': ['weight'], 'quant_bits': {'weight': 8}, 'op_types': ['Linear']}]
    quantizer = NaiveQuantizer(copied_model_naive_quantizer_1, config_list = config_list_naive_quantizer_1)
    masked_model_naive_quantizer_1 = quantizer.compress()
    print(masked_model_naive_quantizer_1)
    test(masked_model_naive_quantizer_1, device, test_loader)

    print('----------------------------------------------------------------------')
    print('Naive Quantizer 2 with 8 bit weight, conv2d operation type:')
    print('Model print output:')
    config_list_naive_quantizer_2 = [{'quant_types': ['weight'], 'quant_bits': {'weight': 8}, 'op_types': ['Conv2d']}]
    quantizer = NaiveQuantizer(copied_model_naive_quantizer_2, config_list = config_list_naive_quantizer_2)
    masked_model_naive_quantizer_2 = quantizer.compress()
    print(masked_model_naive_quantizer_2)
    test(masked_model_naive_quantizer_2, device, test_loader)

    print('----------------------------------------------------------------------')
    print('QAT Quantizer 1 with 8 bit weight and input, Conv2d operation type:')
    print('Model print output:')
    config_list_qat_quantizer_1 = [{'quant_types': ['weight', 'input'], 'quant_bits': {'weight': 8, 'input': 8}, 'op_types': ['Conv2d']}]
    dummy_input = torch.rand(32, 1, 28, 28).to(device)
    quantizer = QAT_Quantizer(copied_model_qat_quantizer_1, config_list_qat_quantizer_1, optimizer, dummy_input)
    masked_model_qat_quantizer_1 = quantizer.compress()
    print(masked_model_qat_quantizer_1)
    test(masked_model_qat_quantizer_1, device, test_loader)

    print('----------------------------------------------------------------------')
    print('QAT Quantizer 2 with 8 bit weight and output, Linear operation type:')
    print('Model print output:')
    config_list_qat_quantizer_2 = [{'quant_types': ['weight', 'output'], 'quant_bits': {'weight': 8, 'output': 8}, 'op_types': ['Linear']}]
    dummy_input = torch.rand(32, 1, 28, 28).to(device)
    quantizer = QAT_Quantizer(copied_model_qat_quantizer_2, config_list_qat_quantizer_2, optimizer, dummy_input)
    masked_model_qat_quantizer_2 = quantizer.compress()
    print(masked_model_qat_quantizer_2)
    test(masked_model_qat_quantizer_2, device, test_loader)

    print('----------------------------------------------------------------------')
    print('DoReFa Quantizer 1 with 8 bit weight and Conv2d operation type:')
    print('Model print output:')
    config_list_dorefa_quantizer_1 = [{'quant_types': ['weight'], 'quant_bits': {'weight': 8}, 'op_types': ['Conv2d']}]
    quantizer = DoReFaQuantizer(copied_model_dorefa_quantizer_1, config_list_dorefa_quantizer_1, optimizer)
    masked_model_dorefa_quantizer_1 = quantizer.compress()
    print(masked_model_dorefa_quantizer_1)
    test(masked_model_dorefa_quantizer_1, device, test_loader)

    print('----------------------------------------------------------------------')
    print('DoReFa Quantizer 2 with 8 bit weight and Linear operation type:')
    print('Model print output:')
    config_list_dorefa_quantizer_2 = [{'quant_types': ['weight'], 'quant_bits': {'weight': 8}, 'op_types': ['Linear']}]
    quantizer = DoReFaQuantizer(copied_model_dorefa_quantizer_2, config_list_dorefa_quantizer_2, optimizer)
    masked_model_dorefa_quantizer_2 = quantizer.compress()
    print(masked_model_dorefa_quantizer_2)
    test(masked_model_dorefa_quantizer_2, device, test_loader)

    #
    # print('----------------------------------------------------------------------')
    # print('L1 Norm Pruned Model with Sparsity 0.8 and Conv2d Operation Type:')
    # print('Model print output:')
    # config_list_l1norm_quantizer_1 = [{'sparsity': 0.8, 'op_types': ['Conv2d']}]
    # l1norm_quantizer_1 = L1Normquantizer(
    #     copied_model_l1norm_quantizer_1, config_list_l1norm_quantizer_1)
    # masked_model_l1norm_quantizer_1, masks = l1norm_quantizer_1.compress()
    # print(masked_model_l1norm_quantizer_1)
    # test(masked_model_l1norm_quantizer_1, device, test_loader)
    #
    # print('----------------------------------------------------------------------')
    # print('L1 Norm Pruned Model with Sparsity 0.2 and Linear Operation Type:')
    # print('Model print output:')
    # config_list_l1norm_quantizer_2 = [{'sparsity': 0.2, 'op_types': ['Linear']}]
    # l1norm_quantizer_2 = L1Normquantizer(
    #     copied_model_l1norm_quantizer_2, config_list_l1norm_quantizer_2)
    # masked_model_l1norm_quantizer_2, masks = l1norm_quantizer_2.compress()
    # print(masked_model_l1norm_quantizer_2)
    # test(masked_model_l1norm_quantizer_2, device, test_loader)
    # print('----------------------------------------------------------------------')

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
