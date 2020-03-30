import numpy as np

from helpers import read_geonames
from ngram_index import NgramIndex

from tqdm import tqdm

tqdm.pandas()

df = read_geonames("allCountries.txt")
df = df[df.feature_class.isin("A P".split())]

ng_index = NgramIndex(n=4)
df["name"].progress_apply(ng_index.split_and_add)

c_codes = df.country_code.unique()
m_count = np.zeros((len(c_codes),len(ng_index.ngram_index)))

countrycode_index = {c_codes[i]:i for i in range(len(c_codes))}

for ix,row in tqdm(df.iterrows(),total=len(df)):
    encoded = ng_index.encode(str(row["name"]))
    country_index = countrycode_index[row.country_code]
    for e in encoded:
        m_count[country_index][e]+=1

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

to_del = [ix for ix,k in ng_index.index_ngram.items() if k.count("$")>2]
m_count = np.delete(m_count,to_del,axis=1)

topn = 20
for ix in range(len(m_count)):
    m_count[ix] = m_count[ix]/np.max(m_count)

sim = cosine_similarity(m_count)

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

plt.imshow(sim)
plt.colorbar()
plt.show()

country_info = pd.read_csv("./country-codes_csv.csv.txt")
iso2toiso3 = dict(country_info[['ISO3166-1-Alpha-2', 'ISO3166-1-Alpha-3']].values)

data = {}
for c_code,c_index in countrycode_index.items():
    if c_code in iso2toiso3:
        data[c_code]=sim[c_index]


world = gpd.read_file("./TM_WORLD_BORDERS_SIMPL-0/TM_WORLD_BORDERS_SIMPL-0.3.shp")
world["sim_vec"] = world.ISO2.apply(lambda x: data[x] if x in data else None)

for c_code in world.ISO2.values:
    sim_vec = world[world.ISO2 == c_code].iloc[0].sim_vec
    try:
        if not sim_vec:
            sim_vec = np.zeros(len(world[world.ISO2 == "FR"].iloc[0].sim_vec))
    except:
        pass
    world["sim_{0}".format(c_code)] = world.ISO2.apply(lambda x : sim_vec[countrycode_index[x]] if x in countrycode_index else 0)


sim_vec = world[world.ISO2 == "FR"].iloc[0].sim_vec
world["simtoselected"] = world.ISO2.apply(lambda x : sim_vec[countrycode_index[x]] if x in countrycode_index else 0)
fig, ax = plt.subplots(1, 1)
world.plot(column='simtoselected', ax=ax, legend=True)
plt.show()

del world["sim_vec"]
world.to_file("simtoponyms.geojson")