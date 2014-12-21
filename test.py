import math, sys
import numpy as np
import gensim.models as models
from gensim import utils, matutils
from gensim.models.word2vec import *
try:
    from gensim_addons.models.word2vec_inner import train_sentence_sg, train_sentence_cbow, train_sentence, FAST_VERSION, train_sentence_test

except ImportError:
    print "IMPORT ERROR"
    def train_sentence(model, sentence_id, sentence, alpha, work=None, neu1=None):
        print "WRONG PLACE"
        for pos, word in enumerate(sentence):
            if word is None:
             continue

            if window!=0:
                reduced_window = random.randint(model.window)
            else:
                reduced_window = 0
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(sentence[start : pos + model.window + 1 - reduced_window], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            l1 = np_sum(model.syn0[word2_indices], axis=0)
            l1 += model.syn0_sen[sentence_id]
            if word2_indices and model.cbow_mean:
                l1 /= (len(word2_indices)+1)
            neu1e = zeros(l1.shape)
            l2a = model.syn1[word.point]
            fa = 1. / (1. + exp(-dot(l1, l2a.T)))
            ga = (1. - word.code - fa) * alpha
            model.syn1[word.point] += outer(ga, l1)
            neu1e += dot(ga, l2a)
            model.syn0[word2_indices] += neu1e
            model.syn0_sen[sentence_id] += neu1e

            return len([word for word in sentence if word is not None])
    def train_sentence_test(model, sentence_id, sentence, alpha, work=None, neu1=None):
        print "WRONG PLACE"
        for pos, word in enumerate(sentence):
            if word is None:
             continue

            if model.window!=0:
                reduced_window = random.randint(model.window)
            else:
                reduce_window = 0
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(sentence[start : pos + model.window + 1 - reduced_window], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            l1 = np_sum(model.syn0[word2_indices], axis=0)
            l1 += model.syn0_sen[sentence_id]
            if word2_indices and model.cbow_mean:
                l1 /= (len(word2_indices)+1)
            neu1e = zeros(l1.shape)
            l2a = model.syn1[word.point]
            fa = 1. / (1. + exp(-dot(l1, l2a.T)))
            ga = (1. - word.code - fa) * alpha
            model.syn1[word.point] += outer(ga, l1)
            neu1e += dot(ga, l2a)
            model.syn0_sen[sentence_id] += neu1e

            return len([word for word in sentence if word is not None])

class Para2Vec(Word2Vec):
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,sample=0, seed=1, workers=4, min_alpha=0.0001, iteration=1):
        self.iteration = iteration
        self.now_iterated = 0
        Word2Vec.__init__(self, sentences=sentences, size=size, alpha=alpha, window=window,
                          min_count=min_count, sample = sample, seed=seed, workers=workers,
                          min_alpha=min_alpha, sg=0,hs=1,negative=0,cbow_mean=0)
    def output(self):
        f = open("words.txt","w")
        for k in self.vocab.keys():
            f.write(k+"\n")
        f.close()

    def build_vocab(self, sentences):
        self.n_sentence = len(sentences)
        Word2Vec.build_vocab(self, sentences)

    def precalc_sampling(self):
        Word2Vec.precalc_sampling(self)

    def create_binary_tree(self):
        Word2Vec.create_binary_tree(self)

    def reset_weights(self):
        print "RESET WEIGHTS...."
        random.seed(self.seed)
        self.syn0 = empty((len(self.vocab), self.layer1_size), dtype=REAL)
                # randomize weights vector by vector, rather than materializing a huge random matrix
                # in RAM at once
        for i in xrange(len(self.vocab)):
            self.syn0[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
        layersize = self.layer1_size * (self.window)
        self.syn1 = empty((len(self.vocab), layersize), dtype=REAL)

        self.syn0_sen = empty((self.n_sentence, self.layer1_size), dtype=REAL)
        for i in xrange(self.n_sentence):
            self.syn0_sen[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
        self.syn0_sen_norm = None
        print "FINISH WEIGHTS...."

    def reset_sentence_array(self, sentences):
        self.n_sentence = len(sentences)
        random.seed(self.seed)
        self.syn0_sen = empty((self.n_sentence, self.layer1_size), dtype=REAL)
        for i in xrange(self.n_sentence):
            self.syn0_sen[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
        self.syn0_sen_norm = None

    def train(self, sentences, total_words=None, word_count=0, chunksize=100):
            for i in range(self.iteration):
                    self.once_train(sentences, total_words,word_count, chunksize)
                    self.now_iterated += 1
    
    def test(self, sentences, total_words=None, word_count = 0, chunksize= 100,iterations=10):
            self.reset_sentence_array(sentences)
            self.iteration = iterations
            self.now_iterated = 0
            for i in range(self.iteration):
                if i % 100==0:
                    print "TESTING : " + str(i) 
                self.once_test(sentences, total_words, word_count, chunksize)
                self.now_iterated +=1
    
    def get_possible(self, sentences, id, word):
        l1 = self.syn0_sen[id]
        try:
            word_index = self.vocab[word]
        except:
            return 0.0
        syn1 = self.syn1[word_index.point]
        codes = word_index.code
        codelen = len(word_index.code)
        t = 1.0
        for i in range(codelen):
            p = dot(syn1[i],l1)
            if codes[i]==0:
                p = -p
            p = self.sigmod(p)
            t = t * p
        return t
    def sigmod(self, x):
        return 1./(1.+math.exp(-x))

    def find_similar(self, sentences, id):
        "print the result"
        for i in range(len(sentences[id])):
            print sentences[id][i]+" ",
        print
        print "---------------------------------------------------------------------"
        res = []
        for words in self.vocab.keys():
            word = self.vocab[words]
            if word.count < 25:
                continue
            similarity = self.sentence_word_similarity(words,id)
            if similarity > 0.3:
                res.append((words, similarity))
        res = sorted(res,cmp=lambda x, y:-cmp(x[1], y[1]))[:40]
        for i in range(len(res)):
            print str(res[i][0])+"\t"+str(res[i][1])
    
    def find_similar_in_sentence(self, sentences, id):
        for i in range(len(sentence[id])):
            print sentence[id][i] +" ",
        print 
        print "--------------------------------------------------------------------"
        res = []
        print sentence[id], set(sentence[id])
        for words in set(sentence[id]):
            similarity = self.get_possible(sentences, id, words)
            #if similarity > 0.3:
            res.append((words, similarity))
        res = sorted(res, cmp=lambda x, y : -cmp(x[1], y[1]))[:40]
        for i in range(len(res)):
            print str(res[i][0]) + "\t" + str(res[i][1])

    def test_wordsim353(self, output_filename):
        my_result = []
        standard_result = []
        with open("wordsim353.test") as f:
            for l in f:
                word1, word2, relation = l.strip().split()
                standard_result.append(float(relation))
                my_result.append(self.similarity(word1, word2))
        with open(output_filename,"a") as fo:
            print type(standard_result[1])
            fo.write(str(self.get_relation(standard_result,my_result))+"\n")
    def similarity(self, w1, w2):
        if w1 not in self:
            return 0.0
        if w2 not in self:
            return 0.0
        return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))
    def save_sentence_vector(self, sentences, output_filename):
        with open(output_filename,"w") as fo:
            for i in range(self.n_sentence):
                #for word in sentences[i]:
                #    fo.write(" " + word)
                #fo.write("\n")
                result = matutils.unitvec(self.syn0_sen[i])
                for j in range(self.layer1_size):
                    fo.write(str(result[j])+" ")
                fo.write("\n")
    def get_relation(self, a, b):
        a = [(x, y) for x, y in enumerate(a)]
        b = [(x, y) for x, y in enumerate(b)]
        a = sorted(a, cmp=lambda x, y:cmp(x[1], y[1]))
        b = sorted(b, cmp=lambda x, y:cmp(x[1], y[1]))
        ranka = [0] * len(a)
        rankb = [0] * len(b)
        for (num, (p, q)) in enumerate(a):
            ranka[p] = num
        for (num, (p, q)) in enumerate(b):
            rankb[p] = num
        mid = 0.5 * (len(a) - 1)
        upper = 0.0
        downer1 = 0.0
        downer2 = 0.0
        for i in range(len(a)):
            upper += (ranka[i] - mid) * (rankb[i] - mid)
            downer1 += (ranka[i] - mid ) * (ranka[i] - mid)
            downer2 += (rankb[i] - mid) * (rankb[i] - mid)
        print ranka, rankb
        return (upper / (math.sqrt(downer1) * math.sqrt(downer2)))

    "just for test"
    def sentence_word_similarity(self, word,sentence_id):
        if word not in self:
            return 0.0
        return dot(matutils.unitvec(self[word]),matutils.unitvec(self.syn0_sen[sentence_id]))

    def once_train(self, sentences, total_words=None, word_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("Cython compilation failed, training will be slow. Do you have Cython installed? `pip install cython`")
        logger.info("training model with %i workers on %i vocabulary and %i features, "
            "using 'skipgram'=%s 'hierarchical softmax'=%s 'subsample'=%s and 'negative sampling'=%s" %
            (self.workers, len(self.vocab), self.layer1_size, self.sg, self.hs, self.sample, self.negative))

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        total_words = total_words or int(sum(v.count * v.sample_probability for v in itervalues(self.vocab)))
        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = zeros(self.layer1_size*self.window, dtype=REAL)  # each thread must have its own work memory
            neu1 = matutils.zeros_aligned(self.layer1_size*self.window, dtype=REAL)

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0*(word_count[0]+self.now_iterated*total_words)/(total_words*self.iteration)))
                # how many words did we train on? out-of-vocabulary (unknown) words do not count

                job_words = sum(train_sentence(self, sentence[0],sentence[1], alpha, work, neu1) for sentence in job)
                with lock:
                    word_count[0] += job_words
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        print "PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %(100.0 *(word_count[0]+self.now_iterated*total_words)/(total_words* self.iteration), alpha, word_count[0] / elapsed if elapsed else 0.0)
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_sentences():
            for number, sentence in enumerate(sentences):
                # avoid calling random_sample() where prob >= 1, to speed things up a little:
                sampled = [self.vocab[word] for word in sentence
                    if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or self.vocab[word].sample_probability >= random.random_sample())]
                sampled = (number, sampled)
                yield sampled


        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(prepare_sentences(), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
            (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))

        return word_count[0]


    def once_test(self, sentences, total_words=None, word_count=0, chunksize=100):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("Cython compilation failed, training will be slow. Do you have Cython installed? `pip install cython`")
        logger.info("training model with %i workers on %i vocabulary and %i features, "
            "using 'skipgram'=%s 'hierarchical softmax'=%s 'subsample'=%s and 'negative sampling'=%s" %
            (self.workers, len(self.vocab), self.layer1_size, self.sg, self.hs, self.sample, self.negative))

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        #total_words = total_words or int(sum(v.count * v.sample_probability for v in itervalues(self.vocab)))
        total_words = 0
        for i in range(len(sentences)):
            total_words += len(sentences[i])

        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of sentences from the jobs queue."""
            work = zeros(self.layer1_size * self.window, dtype=REAL)  # each thread must have its own work memory
            neu1 = matutils.zeros_aligned(self.layer1_size * self.window, dtype=REAL)

            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0*(word_count[0]+self.now_iterated*total_words)/(total_words*self.iteration)))
                # how many words did we train on? out-of-vocabulary (unknown) words do not count

                job_words = sum(train_sentence_test(self, sentence[0],sentence[1], alpha, work, neu1) for sentence in job)
                with lock:
                    word_count[0] += job_words
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        print "PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %(100.0 *(word_count[0]+self.now_iterated*total_words)/(total_words* self.iteration), alpha, word_count[0] / elapsed if elapsed else 0.0)
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        def prepare_sentences():
            for number, sentence in enumerate(sentences):
                # avoid calling random_sample() where prob >= 1, to speed things up a little:
                sampled = [self.vocab[word] for word in sentence
                    if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or self.vocab[word].sample_probability >= random.random_sample())]
                sampled = (number, sampled)
                yield sampled


        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(prepare_sentences(), chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
            (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))

        return word_count[0]






if __name__=="__main__":
    sentences = []
    with open(sys.argv[1]) as f:
        for l in f:
            now_sentence = l.strip().lower().split()
            length = len(now_sentence)
            if length < 9:
                for i in range(length, 9):
                    now_sentence.append("null")
            sentences.append(now_sentence)
    model = Para2Vec(sentences=sentences,workers=24, iteration =1, window=9,
                     size=400,min_alpha=0.00001, min_count=5)
    #print model.similarity("man","woman")
    model.output()
    model.save_sentence_vector(sentences, "para_vectors_train.txt")
    s = []
    with open(sys.argv[2]) as f:
        for l in f:
            s.append(l.strip().lower().split())
    model.test(s,iterations=1)
    model.save_sentence_vector(sentences,"para_vectors_test.txt")
