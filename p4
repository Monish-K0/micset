import pandas as pd
def find_s_algo(file_path):
    data=pd.read_csv(file_path)
    print("\nTrainning data: ",data)
    attributes=data.columns[:-1]
    class_label=data.columns[-1]
    positive=data[data[class_label]=="Yes"]
    hypo=list(positive.iloc[0][attributes])
    for _,row in positive.iterrows():
      for i,value in enumerate(row[attributes]):
          if hypo[i]!=value :
               hypo[i] = "?"
    return hypo
file_path="training_data.csv"
hypo=find_s_algo(file_path)
print("Final Hypothesis: ",hypo)


Sky,AirTemp,Humidity,Wind,Water,Forecast,EnjoySport
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes


