#Importing Lib
from recsys import * ## Recommender system cookbook
from geneneric_preprocessing import * ## pre-processing code
from IPython.display import HTML ## setting display options for IPython Notebook

# Importing ratings data 
ratings = pd.read_csv('./ratings.csv')
ratings.head()
