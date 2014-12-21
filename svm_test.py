#/usr/bin/python
#-*-coding:utf-8-*-

if __name__=="__main__":
    train_filename = "para_train.txt"
    test_filename = "para_test.txt"
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    with open("train_y.txt") as f:
        for l in f:
            train_y.append(float(l.strip()))
    with open("test_y.txt") as f :
        for l in f:
            test_y.append(float(l.strip()))

    with open(train_filename,"r") as f:
        while True:



            l = f.readline()
            if not l :
                break
            l = l.strip().split()
            result = []
            for i in range(len(l)):
                result.append(float(l[i]))
            train_x.append(result)
    print len(train_x)
    print "FINISH LOADING TRAIN"
    with open(test_filename,"r") as f:
        while True:



            l = f.readline()
            if not l :
                break
            l = l.strip().split()
            result = []
            for i in range(len(l)):
                result.append(float(l[i]))
            test_x.append(result)
    print "FINSIH LOADING TEST"
    from sklearn.svm import SVC
    x = SVC()
    x.fit(train_x, train_y)
    print "TRAINING..."
    result = x.predict(test_x)
    print "PREDICTING..."
    num = 0
    for i in range(len(test_y)):
        if test_y[i]!=result[i]:
            num+=1
    print float(num)/float(len(test_y))
    result = x.predict(train_x)
    num = 0
    for i in range(len(train_y)):
        if train_y[i]!=result[i]:
            num+=1
    print float(num)/float(len(train_y))
