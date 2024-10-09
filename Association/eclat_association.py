# Install apyori package
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

dataset = pd.read_csv('Datasets/Market_Basket_Optimisation.csv', header = None)
# Apriori accepts lists
# Convert dataset values to strings while creating transactions
transaction = []
for i in range(len(dataset)):
    transaction.append([str(j) for j in dataset.values[i] if pd.notna(j)])  # Exclude NaN values


#print(transaction)

# Getting rules from apriori

# min support = 3*7/7500, min_confidence = trail and error
# min and max lenth is 2 because our bp requires to find by one get one free relationship
rules = apriori(transactions = transaction, min_support = 0.003, 
                min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

results = list(rules)
print(results)


## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

## Displaying the results non sorted
resultsinDataFrame

## Displaying the results sorted by descending lifts
print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))