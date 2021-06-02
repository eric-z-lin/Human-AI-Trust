import sqlite3 

conn = sqlite3.connect("db.sqlite3") # connect to test.db 
cursor = conn.cursor() # get a cursor to the sqlite database 
# cursor is the object here, you can use any name

# create a table
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# cursor.execute is used to execute any command to sqlite
print(cursor.fetchall())

cursor.execute("SELECT * from mainpage_modeluserresponse")
names = list(map(lambda x: x[0], cursor.description))
print(names)
#print(cursor.fetchall())
import pandas as pd

df = pd.DataFrame(cursor.fetchall(), columns=names)
print(df.head())
df.to_csv('data.csv',index=False)
