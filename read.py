#%%
#read user reviews
import ast
data = []
with open('/Users/chenzichu/Desktop/Capstone/data/steam data/australian_user_reviews.json') as f:
    for line in f:
        data.append(ast.literal_eval(line))

# %%
data[0]['reviews'][0]['review']
# %%
#read steam_reviews
import json, ast
data = []
with open('steam_reviews.json') as f:
    x=1
    for line in f:
        jdata = ast.literal_eval(json.dumps(line)) # Removing uni-code chars
        b=jdata[:-1].replace("u'","'")
        res = ast.literal_eval(b)
        data.append(res)
        x+=1
        print(x)
        if x==10000:
            break


# %%
