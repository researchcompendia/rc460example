setwd(".")
#setwd("~/Dropbox/Duke/2012_computational_adv/LDA_R/")
library(lda)
library(reshape)   # for melt
library(ggplot2)   # for qplot

### LOAD THE DATA ###

data = list()
indi = seq(10,99)
indi_len  = length(indi)

for(i in 1:indi_len){
  app = sprintf("data/analytics/analyst_business_analytics/page_%d.txt",indi[i])
  data[[ 5*(i-1) + 1 ]]= readLines(app)
  
  app = sprintf("data/analytics/data_mining/page_%d.txt",indi[i])
  data[[ 5*(i-1) + 2 ]]= readLines(app)
  
  app = sprintf("data/beauty/hair_loss/page_%d.txt",indi[i])
  data[[ 5*(i-1) + 3 ]]= readLines(app)
  
  app = sprintf("data/fitness/body_tone_mini/page_%d.txt",indi[i])
  data[[ 5*(i-1) + 4 ]]= readLines(app)
  
  app = sprintf("data/fitness/fat_loss/page_%d.txt",indi[i])
  data[[ 5*(i-1) + 5 ]]= readLines(app)
  
  
}

### GENERATE THE CORPUS AND THE VOCABULARY ###

corpus = lexicalize(data,lower=TRUE)
vocab = corpus$vocab[word.counts(corpus$documents, corpus$vocab) > 2]
vocab = sort(vocab)[-seq(1,171)]     # to remove numbers from the vocabulary
vocab = sort(vocab, decreasing=TRUE)[-1]  # to remove zzz
corpus = lexicalize(data,lower=TRUE, vocab=vocab)


### LDA ###

nsamp = 100
niter = 10
pred.prop = matrix(NA,nrow=nsamp*niter,ncol=5)


K = 10 # number of topics

set.seed = 123

for(km in 1:niter){
  result <- lda.collapsed.gibbs.sampler(corpus,
                                        K,    # Num clusters
                                        vocab,
                                        500,  # Num iterations
                                        0.1,  # alpha
                                        0.1,  # eta
                                        compute.log.likelihood=TRUE) 
  
  
  topic.proportions <- t(result$document_sums) / colSums(result$document_sums)
  top.prop <- topic.proportions
  
  
  ### PREDICTION ###
  
  for(j in 1:nsamp){
    x = seq(0,449,by=5)
    av.prop = matrix(NA,nrow=5,ncol=10)
    top.prop.test = matrix(NA,nrow=10*5,ncol=10)
    for(i in 1:5){
      x=x+1
      x.test = sample(x,10)
      top.prop.test[((i-1)*10 +1):((i-1)*10 +10),] = top.prop[x.test,]
      top.prop.est = top.prop[setdiff(x,x.test),]
      av.prop[i,] = colMeans(top.prop.est,na.rm=TRUE)
    }
    
    pred.vect = as.integer(apply(top.prop.test %*% t(av.prop),1,which.max))
    
    for(i in 1:5){
      x = ((i-1)*10 +1):((i-1)*10 +10)
      pred.prop[j+(km-1)*niter,i] = round(length(which(pred.vect[x]==i))/length(pred.vect[x]),2)
    }
  }

}

colnames(pred.prop) <- c("b. analytics","data mining", "hair loss", 
                         "body tone", "fat loss")

### PLOTS ###

#save(pred.prop, file="predprop.Rdata")
boxplot(pred.prop, col="grey", ylab="prediction accuracy", ylim=c(0,1))
