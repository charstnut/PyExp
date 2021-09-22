# This is a notebook for testing and reference
# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import pymongo as pm
import gridfs
import pprint

# %%
# Have to start a mongodb instance (typically through powerscript)
client = pm.MongoClient() # Start a connection
client.database_names() # List all databases

# %%
db_test = client.test # Create or connect to the "test" databse
# coll_dataset = db.create_collection("dataset") # Create a collection
# db_test.drop_collection("dataset") # Drop a collection
db_test.collection_names()

# %%
coll_restaurant = db_test.restaurants # Connect to the collection object

# %%
book = {"Title": "Reference for MongoDB"} # The book is a document in the collection
book ["year"] = 2015
book

# %%
coll_restaurant.insert_one(book)

# %%
cursor = coll_restaurant.find({"address.zipcode": "10075"}) # Return cursor object (like a pandas sheet) with matching findings
for doc in cursor:
    print(doc['_id']) # Print the object id

# %%
coll_restaurant.find_one({"address.zipcode": "10075"})
result = coll_restaurant.update_many({'name': "Nectar Coffee Shop"}, {'$set': {'cuisine': 'American (new)'}, '$currentDate':  {'lastModified': { '$type': "date" }}}) # Filter # add a new filed showing last modified date # modify a field
result.modified_count
coll_restaurant.find_one({"address.zipcode": "10075"})

# %%
coll_restaurant.distinct('Title') # list all the distinct entries in "Title" key
