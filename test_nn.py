import ffnet

if __name__=="__main__":
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(12500):
        test_y.append(1)
        train_y.append(1)
    for i in range(12500):
        test_y.append(0)
        train_y.append(0)
    with open("para_train.txt") as f:
        for l in f:
            l = l.strip().split()
            result = []
            for i in range(len(l)):
                result.append(float(l[i]))
            train_x.append(result)
    print "FINISH READING TRAIN FILE"
    with open("para_test.txt") as f:
        for l in f:
            l = l.strip().split()
            result = []
            for i in range(len(l)):
                result.append(float(l[i]))
            test_x.append(result)
    print "FINISH READING TEST FILE"
    #train_x = train_x[:5]
    #test_x = test_x[:5]
    #train_y = train_y[:5]
    #test_y = test_y[:5]
    c = ffnet.ffnet(ffnet.mlgraph((len(train_x[0]), 50, 1)))
    print "TRAINING....",
    c.train_tnc(train_x, train_y, messages = 1, nproc = 'ncpu', maxfun = 1000)
    print "OK"
    print "TESTING....",
    wrong= 0
    for i in range(len(test_y)):
        result = c.call(test_x[i]).tolist()[0]
        if result >=0.5:
            result = 1.0
        else:
            result = 0.0
        if result != test_y[i]:
            wrong+=1
    print "OK"
    print float(wrong) / float(len(test_y))