import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import deepdish as dd
from fed_script.mlp import MLP

EPS = 1e-15
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Parser:
    def __init__(self):
        self.epochs = 100
        self.lr = 0.001
        self.test_batch_size = 100
        self.batch_size = 40
        self.log_interval = 10
        self.seed = 1
        self.split = 0


args = Parser()
torch.manual_seed(args.seed)

data1 = dd.io.load('/basket/yufeng/data/HO_vector/NYU_correlation_matrix.h5')
data2 = dd.io.load('/basket/yufeng/data/HO_vector/UM_correlation_matrix.h5')


x1 = torch.from_numpy(data1['data']).float()
y1 = torch.from_numpy(data1['label']).long()
x2 = torch.from_numpy(data2['data']).float()
y2 = torch.from_numpy(data2['label']).long()

if False: #save splitting
    n1 = len(y1)//7
    n2 = len(y2)//9
    ll1 = list(range(n1))
    ll2 = list(range(n2))
    random.seed(123)
    random.shuffle(ll1)
    random.shuffle(ll2)
    list1 = dict()
    list2 = dict()
    for i in range(5):
        list1[i] = list()
        if i!=4:
            temp = ll1[i*n1//5:(i+1)*n1//5]
        else:
            temp = ll1[4*n1//5:]
        for j in range(7):
            list1[i]+=list(np.array(temp)*7+j)
    for i in range(5):
        list2[i] = list()
        if i!=4:
            temp = ll2[i*n2//5:(i+1)*n2//5]
        else:
            temp = ll2[4*n2//5:]
        for j in range(9):
            list2[i]+=list(np.array(temp)*9+j)

    dd.io.save('./idx/NYU_sub.h5',{'0':list1[0],'1':list1[1],'2':list1[2],'3':list1[3],'4':list1[4]})
    dd.io.save('./idx/UM_sub.h5',{'0':list2[0],'1':list2[1],'2':list2[2],'3':list2[3],'4':list2[4]})


idNYU = dd.io.load('./idx/NYU_sub.h5')
idUM = dd.io.load('./idx/UM_sub.h5')

if args.split==0:
    tr1 = idNYU['1']+idNYU['2']+idNYU['3']+idNYU['4']
    tr2 = idUM['1']+idUM['2']+idUM['3']+idUM['4']
    te1=  idNYU['0']
    te2 = idUM['0']
elif args.split==1:
    tr1 = idNYU['0']+idNYU['2']+idNYU['3']+idNYU['4']
    tr2 = idUM['0']+idUM['2']+idUM['3']+idUM['4']
    te1=  idNYU['1']
    te2 = idUM['1']
elif args.split==2:
    tr1 = idNYU['0']+idNYU['1']+idNYU['3']+idNYU['4']
    tr2 = idUM['0']+idUM['1']+idUM['3']+idUM['4']
    te1=  idNYU['2']
    te2 = idUM['2']
elif args.split==3:
    tr1 = idNYU['0']+idNYU['1']+idNYU['2']+idNYU['4']
    tr2 = idUM['0']+idUM['1']+idUM['2']+idUM['4']
    te1=  idNYU['3']
    te2 = idUM['3']
elif args.split==4:
    tr1 = idNYU['0']+idNYU['1']+idNYU['2']+idNYU['3']
    tr2 = idUM['0']+idUM['1']+idUM['2']+idUM['3']
    te1=  idNYU['4']
    te2 = idUM['4']

x1_train = x1[tr1]
y1_train = y1[tr1]
x2_train = x2[tr2]
y2_train = y2[tr2]
x1_test = x1[te1]
y1_test = y1[te1]
x2_test = x2[te2]
y2_test = y2[te2]


mean = x1_train.mean(0, keepdim=True)
dev = x1_train.std(0, keepdim=True)
x1_train = (x1_train - mean) / dev
x1_test = (x1_test - mean) / dev

mean = x2_train.mean(0, keepdim=True)
dev = x2_train.std(0, keepdim=True)
x2_train = (x2_train - mean) / dev
x2_test = (x2_test - mean) / dev


train1 = TensorDataset(x1_train, y1_train)
train_loader1 = DataLoader(train1, batch_size=args.batch_size, shuffle=True)

train2 = TensorDataset(x2_train, y2_train)
train_loader2 = DataLoader(train2, batch_size=args.batch_size, shuffle=True)


test1 = TensorDataset(x1_test, y1_test)
test_loader1 = DataLoader(test1, batch_size=args.test_batch_size, shuffle=True)

test2 = TensorDataset(x2_test, y2_test)
test_loader2 = DataLoader(test2, batch_size=args.test_batch_size, shuffle=True)


model = MLP(6105,16,2)
optimizer1 = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-2)
print(model)
nnloss = nn.NLLLoss()



def train(data_loader,optimizer,epoch,model):
    model = model.to(device)
    model.train()
    if epoch <= 50 and epoch % 20 == 0:
        for param_group1 in optimizer1.param_groups:
            param_group1['lr'] = 0.5 * param_group1['lr']
    elif epoch > 50 and epoch % 20 == 0:
        for param_group1 in optimizer1.param_groups:
            param_group1['lr'] = 0.5 * param_group1['lr']

    loss_all1 = 0

    for data, target in data_loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output1 = model(data)
        loss1 = nnloss(output1, target)
        loss1.backward()
        loss_all1 += loss1.item() * target.size(0)
        optimizer.step()


    return loss_all1 / (len(data_loader.dataset)), model


def test(federated_model,test_loader):
    federated_model = federated_model.to(device)
    federated_model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = federated_model(data)
        test_loss += nnloss(output, target).item() * target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Average acc: {:.4f}'.format(test_loss,correct))


for epoch in range(args.epochs):
    start_time = time.time()
    print(f"Epoch Number {epoch + 1}")
    l1,federated_model = train(train_loader2,optimizer1,epoch,model)
    print(' L1 loss: {:.4f}'.format(l1))
    test(federated_model,test_loader2)
    total_time = time.time() - start_time
    print('Communication time over the network', round(total_time, 2), 's\n')

print('split:', args.split )