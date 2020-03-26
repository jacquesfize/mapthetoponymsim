# Map the similarities between countries toponyms 

(in progress)
In this project, I propose a python script to compute the similarity between countries' toponyms. I'm using the Geonames dataset to collect toponyms. Each toponym is encoded in a sequence of 4-gram (e.g. Paris --> {$$$P,$$Pa,$Par,Pari,...}). For each country, we produce a vector that keep counts of ngram frequency. Similarity is then computed using the cosine similarity function.



![MX_example](example_MX.png)
<p style="text-align:center">Example of map generated on Mexico</p>
# Clone the git repo

    $ git clone https://github.com/Jacobe2169/mapthetoponymsim.git

# First Requirements

    $ pip3 install -r requirements.txt

# Download Data

    * Geonames data : http://download.geonames.org/export/dump/allCountries.zip 
    * Countries borders shape : https://thematicmapping.org/downloads/TM_WORLD_BORDERS-0.3.zip

Then, unzip both archive in the script directory.