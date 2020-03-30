import json

import numpy as np

from ngram import NGram


class NgramIndex():
    """
    Class used for encoding words in ngram representation
    """
    def __init__(self,n,loaded = False):
        """
        Constructor
        
        Parameters
        ----------
        n : int
            ngram size
        """
        self.ngram_gen = NGram(N=n)

        self.size = n
        self.ngram_index = {"":0}
        self.index_ngram = {0:""}
        self.cpt = 0
        self.max_len = 0

        self.loaded = loaded
    def split_and_add(self,word):
        """
        Split word in multiple ngram and add each one of them to the index
        
        Parameters
        ----------
        word : str
            a word
        """
        ngrams = word.lower().replace(" ","$")
        ngrams = list(self.ngram_gen.split(ngrams))
        [self.add(ngram) for ngram in ngrams]
        self.max_len = max(self.max_len,len(ngrams))

    def add(self,ngram):
        """
        Add a ngram to the index
        
        Parameters
        ----------
        ngram : str
            ngram
        """
        if not ngram in self.ngram_index:
            self.cpt+=1
            self.ngram_index[ngram]=self.cpt
            self.index_ngram[self.cpt]=ngram
        

    def encode(self,word):
        """
        Return a ngram representation of a word
        
        Parameters
        ----------
        word : str
            a word
        
        Returns
        -------
        list of int
            listfrom shapely.geometry import Point,box
 of ngram index
        """
        ngrams = word.lower().replace(" ","$")
        ngrams = list(self.ngram_gen.split(ngrams))
        return [self.ngram_index[ng] for ng in ngrams if ng in self.ngram_index]

    def complete(self,ngram_encoding,MAX_LEN,filling_item=0):
        """
        Complete a ngram encoded version of word with void ngram. It's necessary for neural network.
        
        Parameters
        ----------
        ngram_encoding : list of int
            first encoding of a word
        MAX_LEN : int
            desired length of the encoding
        filling_item : int, optional
            ngram index you wish to use, by default 0
        
        Returns
        -------
        list of int
            list of ngram index
        """
        if self.loaded and len(ngram_encoding) >=MAX_LEN:
            return ngram_encoding[:MAX_LEN]
        assert len(ngram_encoding) <= MAX_LEN
        diff = MAX_LEN - len(ngram_encoding)
        ngram_encoding.extend([filling_item]*diff)  
        return ngram_encoding

    def save(self,fn):
        """

        Save the NgramIndex
        
        Parameters
        ----------
        fn : str
            output filename
        """
        data = {
            "ngram_size": self.size,
            "ngram_index": self.ngram_index,
            "cpt_state": self.cpt,
            "max_len_state": self.max_len
        }
        json.dump(data,open(fn,'w'))

    @staticmethod
    def load(fn):
        """
        
        Load a NgramIndex state from a file.
        
        Parameters
        ----------
        fn : str
            input filename
        
        Returns
        -------
        NgramIndex
            ngram index
        
        Raises
        ------
        KeyError
            raised if a required field does not appear in the input file
        """
        try:
            data = json.load(open(fn))
        except json.JSONDecodeError:
            print("Data file must be a JSON")
        for key in ["ngram_size","ngram_index","cpt_state","max_len_state"]:
            if not key in data:
                raise KeyError("{0} field cannot be found in given file".format(key))
        new_obj = NgramIndex(data["ngram_size"],loaded=True)
        new_obj.ngram_index = data["ngram_index"]
        new_obj.index_ngram = {v:k for k,v in new_obj.ngram_index.items()}
        new_obj.cpt = data["cpt_state"]
        new_obj.max_len = data["max_len_state"]
        return new_obj

