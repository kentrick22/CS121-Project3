'''
Group 44:
Keslin Phan
UCInetID: keslinp

Edriana Tanowidjaja
UCInetID: etanowid

Kentrick Kepawitono
UCInetID: kkepawit
'''
from index_constructor import *
from basic_query import *
from gui import *
import json



if __name__ == '__main__':
    bookkeeping = {}
    doc_length = {}


    #load bookkeeping.json and store it in dictionary called bookkeeping
    with open('./WEBPAGES_RAW/bookkeeping.json') as bkfile:
        bookkeeping = json.load(bkfile)


    '''comment line 32, 33 to go straight to query search'''

    # Building index
    # constructIndex(bookkeeping)
    # generateJSONFile()


    #load doc_length.json and store it in dictionary called doc_length to be used during query
    with open('doc_length.json') as dlfile:
        doc_length = json.load(dlfile)


    # Search query
    startGUI(doc_length, bookkeeping)

    print("boo byee :)")
