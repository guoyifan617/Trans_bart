#encoding: utf-8

import torch
from Spreader import SpreaderFunc

a = torch.randn(5,requires_grad=True)
b = torch.randn(2,3,5,requires_grad=True)
c = torch.randn(2,5,requires_grad=True)
rs = SpreaderFunc(a.sigmoid(), b.clone(), c, 1, True)
rs.sum().backward()
ag=a.grad.clone()
bg=b.grad.clone()
cg=c.grad.clone()
a.grad.zero_()
b.grad.zero_()
c.grad.zero_()
rsl = []
for i in range(b.size(1)):
	if i == 0:
		rsl.append(c*a.sigmoid()+b.select(1,0))
	else:
		rsl.append(rsl[-1]*a.sigmoid()+b.select(1,i))
rsl=torch.stack(rsl,1)
print(rs)
print(rsl)
rsl.sum().backward()
print(ag)
print(a.grad)
print(bg)
print(b.grad)
print(cg)
print(c.grad)
