from datapre.preprocessTools import loadTrainingDataFeatures

# Load feature matrix

idlist, X = loadTrainingDataFeatures()
print(X)
print(idlist)