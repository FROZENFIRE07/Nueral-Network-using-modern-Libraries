import random
import torch
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt

data=[]
for _ in range(1000):
    a=random.uniform(0,1)
    b=random.uniform(0,1)
    x=[a,b]
    label=1 if a>b else 0
    data.append((x,label))

x_list=[]
y_list=[]

for pair in data:
    x_list.append(pair[0])
    y_list.append(pair[1])

x=torch.tensor(x_list,dtype=torch.float32)
y=torch.tensor(y_list,dtype=torch.long)

class GreaterNet(nn.Module):
    def __init__(self):
        super(GreaterNet,self).__init__()
        self.hidden=nn.Linear(2,4)
        self.output=nn.Linear(4,2)
    def forward(self,x):
        x=torch.tanh(self.hidden(x))
        x=self.output(x)
        return x

model=GreaterNet()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.4)


for epoch in range(2000):
    outputs=model(x)
    loss=criterion(outputs,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0 :
        print(f"Epoch{epoch},Loss:{loss.item():.6f}")

a = float(input("Enter first number: "))
b = float(input("Enter second number: "))

test_input = torch.tensor([[a, b]], dtype=torch.float32)

with torch.no_grad():
    output = model(test_input)
    predicted = torch.argmax(output).item()

if predicted==1:
    greater_num=a
else:
    greater_num=b

print("Prediction:", predicted," ",greater_num,"is greater")

outputs = model(x)
loss = criterion(outputs, y)
make_dot(loss, params=dict(model.named_parameters())).render("final_computation_graph", format="png")


