import model
import torch
from torch.autograd import Variable

net = model.GoNet()
x = Variable(torch.randn(1,3,227,227))
y = Variable(torch.randn(1,3,227,227))
z = net(x,y)

g = Variable(torch.randn(1,4), requires_grad=False)
loss_fn = torch.nn.MSELoss(size_average=False)
loss = loss_fn(z, g)
loss.backward()

print net.classifier[9]
print net.classifier[9].weight.grad.data

print net.features[0]
try:
    print net.features[0].weight.grad.data
except:
    print('No grads found! Which implies frozen weights')

