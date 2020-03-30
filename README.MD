# Map the similarities between countries toponyms 

This repository contains a code to compute (then map) similarities between countries' place names , or **toponyms**. In this experiment, we collect toponyms from the [Geonames dataset](http://www.geonames.org). To compute the similarity, toponyms are encoded in a character n-gram sequence (e.g. Paris --> {$$$P,$$Pa,$Par,Pari,...}). Character ngram are widely known to be flexible in terms if spelling errors. Here, such representation is used to capture the association between an affix and specific region. For example, the prefix "tre" is commonly used in *Bretagne* (region in France) because it means "populated places" in the local language. 

Once toponyms are encoded, we produce a vector that keep counts of ngram occurrences in each country toponyms (basically a Bag-of-Words!). Finally, the cosine similarity is used to compute the similarity between these vectors.

![MX_example](example_MX.png)

<p align="center">Example of map generated on Mexico</p>

<hr>

# Installation 

## Clone

First, you need to clone the code repository using :

    $ git clone https://github.com/Jacobe2169/mapthetoponymsim.git

## Requirements

This script was conceived in Python3.6 and requires specific libraries. Install the requirements using the following command in your terminal:

    $ pip3 install -r requirements.txt

## Download Data
You need to download the following dataset :

 * Geonames data : http://download.geonames.org/export/dump/allCountries.zip 
 * Countries borders shape : https://thematicmapping.org/downloads/TM_WORLD_BORDERS-0.3.zip

Then, unzip both archive in the script directory.

<hr>

# Run the script

Use the following command:

    python3 map.py

<hr>
# Authors

Developped by Jacques Fize