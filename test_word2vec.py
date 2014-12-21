import gensim
s = []
with open("../text8") as f:
	for l in f:
		s.append(l.strip().split())

w = gensim.models.Word2Vec(s,workers=24)
print w.similarity("man","woman")		
