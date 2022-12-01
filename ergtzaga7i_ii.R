#task7i

calculateRMSE<-function(predictedValues, actualValues){
  err<- sqrt( mean((actualValues - predictedValues)^2) )
  return( err )
}

kFoldCrossValidation<-function(data, frml, k){
  dataset<-data[sample(nrow(data)),]
  folds <- cut(seq(1,nrow(dataset)), breaks=k, labels=FALSE)
  RMSE<-vector()
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    candidate.linear.model<-lm( frml, data = trainData)
    predicted<-predict(candidate.linear.model, testData)
    error<-calculateRMSE(predicted, testData[, "area"])
    RMSE<-c(RMSE, error)
  }
  return( mean(RMSE) )
}


forestfires<-read.csv("C:/Users/Downloads/forestfires.csv", sep=",", header=T, stringsAsFactors = F, quote = "\"")


predictionModel<-vector()
predictionModel<-"area ~ temp+wind+rain"
modelMeanRMSE<-vector()

for (k in 1:length(predictionModel)){
  modelErr<-kFoldCrossValidation(forestfires, as.formula(predictionModel), 10)
  modelMeanRMSE<-c(modelMeanRMSE, modelErr)
}
print( sprintf("Model with best accuracy was: [%s] error: [%f]", 
  predictionModel, modelMeanRMSE) )
  
#task7ii

subforestfires <- forestfires[which(forestfires$area<3.2),]
modelErr<-kFoldCrossValidation(subforestfires, as.formula(predictionModel), 10)

modelMeanRMSE<-c(modelMeanRMSE, modelErr)
print( sprintf("Linear regression model [%s]: prediction error [%f]", predictionModel, modelErr ) )

