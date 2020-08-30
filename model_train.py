import torch

def train(epoch, net, trainDataLoader, optimizer, criterion, validDataLoader):

    net.train()
    train_loss = 0
    for sample in trainDataLoader:

        inputs, targets = sample['X'], sample['Y']
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()   
        train_loss += loss.item()
    
    if epoch % 10 == 0:

        net.eval()
        valid_loss = 0
        for sample in validDataLoader:

            inputs, targets = sample['X'], sample['Y']
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()

        # calculate average losses
        train_loss = train_loss/len(trainDataLoader.sampler)
        valid_loss = valid_loss/len(validDataLoader.sampler)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

def test(net, testDataLoader):
    
    net.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for sample in testDataLoader:

            inputs, targets = sample['X'], sample['Y']
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
                
            outputs = net(inputs)
            _, pred = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (pred == targets).sum().item()

    print('\nAccuracy of the network on test: %d %%' % (100 * correct / total))
