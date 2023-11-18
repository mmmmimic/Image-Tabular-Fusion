from transformers import TapexTokenizer, BartForSequenceClassification
import pandas as pd
import json

tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-tabfact", cache_dir='./')
model = BartForSequenceClassification.from_pretrained("microsoft/tapex-large-finetuned-tabfact", cache_dir='./')

data = {
    "wheelbase size": [2606],
    "height": [1421],
    "width": [1786],
    "length": [4367],
    "advertisement year": [2018],
    "advertisement month": [8],
    "registration year": [2006],
    "miles driven": [94000],
    "price": [2500],
    "number of seats": [4],
    "number of doors": [4],
    "original price": [19182],
    "engine size": [2],
    "color": ["Silver"],
    "body type": ['Convertible'],
    'gearbox type': ['Manual'],
    'fuel type': ['petrol']
}


# data = {
#   "sex": ['male'],
#   'race':['white'],
#   'zip code': [720],
#   'extensive burns': ['no'],
#   'malnutrition': ['yes'],
#   'currently pregnant': ['no'],
#   'chronic kidney disease': ['yes'],
#   'diabetes type I': ['no'],
#   'diabetes type II': ['yes'],
#   'transplant surgery': ['no'],
#   'hemodialysis pre diagnosis': ['no'],
#   'cancer': ['yes'],
#   'age': [67],
#   'latest bmi':[19], 
#   'latest weight': [121],
#   'latest height': ['170.1']  
# }

table = pd.DataFrame.from_dict(data)

# tapex accepts uncased input since it is pre-trained on the uncased corpus
# dvm-car
# query = "The car has a wheelbase size of 2606.0. Its height is 1421.0 and its width measures 1786.0. The car's length is 4367.0. It was advertised in August 2018 and registered in 2006. The car has been driven for approximately 94,000 miles. It is priced at $2,500. The car has 4 seats and 2 doors. The original price of the car was $19,182. The engine size is 2.0 and the color of the car is Silver. The body type is Convertible and it has a Manual gearbox. The car runs on Petrol fuel."
# query = "This car has a wheelbase size of 2606.0, a height of 1421.0, a width of 1786.0, and a length of 4367.0. It was advertised in August 2018 and registered in 2006. The car has been driven for approximately 94,000 miles and is priced at $2,500. It has 4 seats and 2 doors. The original price of the car was $19,182. The engine size is 2.0 and the color is Silver. The body type is Convertible and the gearbox type is Manual. It runs on Petrol fuel."

# query = "This car is a silver convertible with a manual gearbox. The car has a wheelbase size of 2606 mm, a height of 1421 mm, a width of 1786 mm, and a length of 4367 mm. The car was advertised in August, 2018 and registered in 2006. The car has been driven for 94000 miles and is priced at 2500. Tha car has 4 seats and 4 doors. The original price of the car was 19182. The engine size of the car is 2 and it runs on petrol fuel."

query = "The car has been driven for 94000 miles and is priced at 2500. The car has 4 seats and 4 doors. The original price of the car was 19182. The engine size of the car is 2 and it runs on petrol fuel."
# query = "The car has been driven for 94000 miles and is priced at 2500. The original price of the car was 19182. The engine size of the car is 2 and it runs on petrol fuel."

# covid19-ar
# query = ['The patient is a 67-year-old male of white race residing in zip code 720.', 'The patient does not have extensive burns and has malnutrition, and is not currently pregnant.', 'The patient is diagnosed with chronic kidney disease.', 'The patient does not have type I diabetes and has type II diabetes.', 'The patient does not undergone any transplant surgeries and the patient does not have hemodialysis prior to the COVID-19 diagnosis.', 'The patient has been diagnosed with cancer.', "The patient's latest BMI is 19.0, with a weight of 121.0 pounds and a height of 170.1 cm."]
# query = ' '.join(query)
# query = "The COVID-19 patient is a 51-year-old male with a weight of 45 kilograms."
# query = "The male COVID-19 patient who is 51-year-old has a weight of 45 kilograms."


# isup

encoding = tokenizer(table=table, query=query, return_tensors="pt")

outputs = model(**encoding)
print(outputs.logits[0].softmax(dim=0))
output_id = int(outputs.logits[0].argmax(dim=0))

print(model.config.id2label[output_id])