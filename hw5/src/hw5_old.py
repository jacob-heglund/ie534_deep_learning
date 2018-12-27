
# this version is old and may not work
def train():
    best_model = {}
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        print('\n------------Training------------\n')
        model.train()
        time1 = time.time()
        for i, (query, pos, neg) in enumerate(train_loader):
            query = query.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            
            lrSchedule(epoch, decay_epoch, scaling ,optimizer, learning_rate)
            optimizer.zero_grad()
            
            # Forward pass
            query_embed = model.forward(query)
            pos_embed = model.forward(pos)
            neg_embed = model.forward(neg)
            
            
           # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/loss.py
           ## triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            #input1 = torch.randn(100, 128, requires_grad=True)
            #input2 = torch.randn(100, 128, requires_grad=True)
            #input3 = torch.randn(100, 128, requires_grad=True)
            #output = triplet_loss(input1, input2, input3)
            #output.backward()
            
            loss.backward()
            # avoid numerical issues
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if('step' in state and state['step']>=1024):
                        state['step'] = 1000

            optimizer.step()
            
            if i % 100 == 0:
                print ("Epoch: [{}/{}] ----- Step: [{}/{}] ----- Percent Complete: {:.4f} ----- Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i, numTrainSteps, i / numTrainSteps, loss.item()))
        
        time2 = time.time()
        print('Epoch Runtime: {} seconds'.format(time2-time1))
        trainLoss[epoch] = float(loss)    
        
        print('\n------------Validation------------\n')
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in testLoader:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            currTestAcc = 100 * (correct / total)
            print('Validation Accuracy: {} %'.format(currTestAcc))
        testAcc[epoch] = currTestAcc

        if testAcc[epoch] > best_test_acc:
            best_model = model.state_dict()
            torch.save(best_model, 'best_model.pt')

