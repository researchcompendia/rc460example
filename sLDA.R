#setwd("~/Dropbox/Duke/2012_computational_adv/sLDA_R/")
setwd(".")
library(lda)
library(reshape)   # for melt
library(ggplot2)   # for qplot


### LOAD THE DATA ###

data = list()
indi = seq(10,99)
indi_len  = length(indi)
annotations = rep(NA,5*indi_len)


for(i in 1:indi_len){
  app = sprintf("data/analytics/analyst_business_analytics/page_%d.txt",indi[i])
  data[[ 5*(i-1) + 1 ]]= readLines(app)
  annotations[5*(i-1) + 1] = rbinom(1,1,1)
  
  app = sprintf("data/analytics/data_mining/page_%d.txt",indi[i])
  data[[ 5*(i-1) + 2 ]]= readLines(app)
  annotations[5*(i-1) + 2] = rbinom(1,1,1)  
  
  app = sprintf("data/beauty/hair_loss/page_%d.txt",indi[i])
  data[[ 5*(i-1) + 3 ]]= readLines(app)
  annotations[5*(i-1) + 3] = rbinom(1,1,0)  
  
  app = sprintf("data/fitness/body_tone_mini/page_%d.txt",indi[i])
  data[[ 5*(i-1) + 4 ]]= readLines(app)
  annotations[5*(i-1) + 4] = rbinom(1,1,0)  
  
  app = sprintf("data/fitness/fat_loss/page_%d.txt",indi[i])
  data[[ 5*(i-1) + 5 ]]= readLines(app)
  annotations[5*(i-1) + 5] = rbinom(1,1,0)  
  
}

### GENERATE THE CORPUS AND THE VOCABULARY ####

corpus = lexicalize(data,lower=TRUE)
vocab = corpus$vocab[word.counts(corpus$documents, corpus$vocab) > 2]
vocab = sort(vocab)[-seq(1,171)]     # to remove numbers from the vocabulary
vocab = sort(vocab, decreasing=TRUE)[-1]  # to remove zzz
corpus = lexicalize(data,lower=TRUE, vocab=vocab)

idx.len = rep(0,indi_len*5)
for( i in 1:(indi_len*5)){
  if(ncol(corpus[i][[1]])<5){
    idx.len[i]=1
  }
}
corpus2 = corpus[-which(idx.len==1)]
annotations = annotations[-which(idx.len==1)]


### sLDA ###

niter = 10
nsamp = 180
tot.corpus = length(annotations)
K = 5 # number of topics
# set.seed = 123

pr = {}
label = {}

for(km in 1:niter){
  print(km)
  
  idx = sample(tot.corpus,nsamp)
  corpus3 = corpus2[-idx]
  annotations3 = annotations[-idx]
  corpus.out = corpus2[idx]
  annotations.out = annotations[idx]
  
  result <- slda.em(corpus3, K, vocab, num.e.iterations=20, num.m.iterations=5, alpha=0.1,
                  eta=0.1, annotations3, params=rep(0,K), logistic = TRUE, variance = 1,
                  regularise = FALSE, method = "sLDA")
  
  predictions <- slda.predict(corpus.out, result$topics, result$model, alpha = 0.1, eta=0.1)
  
  
  pr = c(pr,exp(predictions)/(1 + exp(predictions)))
  label = c(label,annotations.out)
}
  
#save(pr,file="pr.Rdata")
#save(label, file="label.Rdata")

#load("pr.Rdata")
#load("label.Rdata")

### PLOT ###

par(mfrow=c(1,1))
boxplot(pr~label, col="grey", xlab="group", ylab="success probability")

id = which(label==0)
mean(pr[id]<0.5, na.rm=TRUE) +
mean(pr[-id]>0.5, na.rm=TRUE)
