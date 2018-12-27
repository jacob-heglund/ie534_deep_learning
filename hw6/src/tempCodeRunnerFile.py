dx, (X_batch, Y_batch) = testloader.next()
    X_batch = Variable(X_batch,requires_grad=True).cuda()
    Y_batch_alternate = (Y_batch + 1)%10
    Y_batch_alternate = Variable(Y_batch_alternate).cuda()
    Y_batch = Variable(Y_batch).cuda()


    ## save real images
    samples = X_batch.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples[0:100])
    plt.savefig('visualization/real_images.png', bbox_inches='tight')
    plt.close(fig)