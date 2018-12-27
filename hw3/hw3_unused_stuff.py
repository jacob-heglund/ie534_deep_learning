class cifarDataset():
    # this function runs when the class is instantiated
    # use to load the data from disk into numpy arrays
    def __init__(self):
        print("\nLoading Data\n")
        # taken from https://www.cs.toronto.edu/~kriz/cifar.html
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        # gets a list of class labels
        # ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        #self.classes = unpickle('C:\home\classes\IE534_DL\hw3\data\cifar-10-batches-py\\batches.meta')
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # get a numpy arrays containing all the data
        # each batch is a dict with keys:
        # b'batch_label', b'labels', b'data', b'filenames'
        xTrain = np.zeros([1, 3072])
        yTrain = np.zeros(1)
        
        # go through each file
        for i in range(5):
            fileName = 'data_batch_' + str(i+1)
            dataPath = "C:\home\classes\IE534_DL\hw3\data\cifar-10-batches-py\\" + fileName
            print(fileName)
            batch =  unpickle(dataPath)
            # labelTrain is 10000 x 1
            labelTrain = batch[b'labels']
            
            # data is 10000 x 3072 
            dataTrain =  batch[b'data']

            xTrain = np.vstack((xTrain, dataTrain))
            yTrain = np.hstack((yTrain, labelTrain))

        xTrain = np.delete(xTrain, 0, 0)
        yTrain = np.delete(yTrain, 0, 0)

        batchTest = unpickle('C:\home\classes\IE534_DL\hw3\data\cifar-10-batches-py\\test_batch')
        xTest = batchTest[b'data']
        yTest = batchTest[b'labels']

        # yTrain is a list for some reason
        yTest = np.asarray(yTest)

        # convert to pytorch tensors on GPU if available, otherwise on CPU
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        self.xTrain = torch.from_numpy(xTrain).to(device)
        self.yTrain = torch.from_numpy(yTrain).to(device)
        self.xTest = torch.from_numpy(xTest).to(device)
        self.yTest = torch.from_numpy(yTest).to(device)

        print("\nData Loaded\n")

    # return the data as pytorch tensors
    def getData(self):
        return self.xTrain, self.yTrain, self.xTest, self.yTest, self.classes


#dataset = cifarDataset()
#xTrain, yTrain, xTest, yTest, classes = dataset.getData()
