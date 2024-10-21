from predict import predict
from train import train

train("input/trainData.csv", "output/model.bin")
predict("output/model.bin", "input/futureClimateData.csv", "output/predictions.csv")