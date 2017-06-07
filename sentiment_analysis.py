"""
seem fyp: text information mining - aspect based sentiment anaylsis on customer reviews
"""
__author__ = 'Lee Kin Shing'

import sys, os, re, string, math, datetime, random, pdb, pickle
from nltk.tokenize import sent_tokenize, RegexpTokenizer, word_tokenize
from autocorrect import spell
from pycorenlp import StanfordCoreNLP
import networkx as nx
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
import kmedoids
from matplotlib import pyplot as plt
from nltk.corpus import sentiwordnet as swn
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

min_time_thr = datetime.datetime(1995,1,1) 
max_time_thr = datetime.datetime(2016,5,1)
candidates_count = 0
error = 0
occ_thr = 0
setsize = 0
k = 0
source = outdir = None
kmeans = None
model = None
bigram = None
trigram = None
reviews = trainset = full_text = corpus = word_corpus = features = temp_score_array = score_array = clusters = filtered_candidates_key = []
stopword = []
candidates = []
candidates_by_review = full_wordlist = corpus_wordlist = clusters_dict = clusters_for_candidates = final_candidates = {}
min_time = datetime.datetime(2900, 1, 1)
max_time = datetime.datetime(1900, 1, 1)
wnl = WordNetLemmatizer()
G = nx.MultiGraph()
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 50000
nlp = StanfordCoreNLP('http://localhost:9000')


"""
reviews[]: stores Review objects e.g. reviews[] = [Review object 1, Review obejct 2, ...]
trainset[]: stores the the index of reviews that are trainset e.g. trainset = [2, 4, 6, 8, 9, 15, ...]
full_text[]: 2D array to store each tokenized sentence of each review e.g. full_text[] = [["review0sent0", "review0sent1",...], ...]
corpus[]: same as full_text[] but corpus[] only stores reviews that are in trainset e.g. corpus[] = [["review2sent0", "review2sent1",...], ...]
candidates[]: dictionary that stores each LEMMATIZED feature and its opinion words as well as their corresponding appearance
                e.g. candidates{} = {'phone': [['good', 'nice'], [(0, 2), (5,8)], ...]}, 0 here is the 0-th review in corpus[] or trainset[]
                but not the one in full_text[], where in this example 0 is the 2nd review in full_text[]
full_wordlist{}: dictionary that stores each of the word in the entire corpus of reviews and their corresponding occurrence count, note that
                 the word in this wordlist are STEMMED
                 e.g. full_wordlist{} = {'realli': 5, 'good': 17, ...}
corpus_wordlist{}: same as full_wordlist{} but corpus_wordlist{} only stores reviews that are in trainset
features[]: a SORTED list that stores every LEMMATIZED(for finding similarity score ) features generated from previous stages with duplicates
            e.g. features = ['phone', 'phone', ..., 'product', 'product', ...]
word_corpus[]: a list that stores list of tokenized sentences without regard to what review it belongs to
               e.g. word_corpus[] = [['this', 'is', 'a', 'sent'], [...], ...]
temp_score_array[]: same as score_array[], but it is used to transform to np array
score_array[]: np array that stores the similarity score vector of each feature     score_array = [[0.1234,...], [...]]
clusters[]: stores cluster labels e.g. clusters = [1, 2, 0.111, ...], which means the first feature in the features list has cluster label 1, and
            so on, it has SAME len as the len of features[]
clusters_dict{}: dictionary that stores labels and their corresponding group of features
                 e.g. clusters_dict{} = {0: ['car', 'vehicle', ...], 1: ['phone', 'cellphone', ...], ...}
clusters_for_candidates{}: dictionary that stores the candidates and the candidates' info in each clusters,
                           e.g. clusters_for_candidates = {0: [..., ['phone', 'good', 10, 19, 0.74], ['phone', 'original', 0, 23, 0.001], ...], 1: [...], ...},
                                the candidates in a cluster are stored by their sentiment score
candidates_by_review{}: dictionary that stores candidates by reviews, the key is the corpus index, the value is list of candidates, it is for final
                        output e.g. candidates_by_review{} = {0: [['good phone', 11], ['very bad product', 2]], 1: ..., ...}
filtered_candidates_key[]: stores candidates and there corresponding information that are to be filtered
                           e.g. filtered_candidates_key[] = [['phone', 'original', 0.01, 11, 23], ...]
final_candidates[]: same as candidates but with filtered
"""


class Review:
    """data structure of a single reivew object"""
    """contains user, product, rating and features for a single review"""
    def __init__(self, user, product, rating, time=-1, ntime=-1):
        self.user = user
        self.product = product
        self.rating = rating
        self.features = []
        self.time = time
        self.ntime = ntime

    def set_ntime(self, time_span_days):
        self.ntime = (self.time - min_time).days / time_span_days
        return self.ntime
     
    def set_features(self, features):
        self.features += features
        return self.features

    def to_str(self, features = True):
        ntime = self.ntime
        if ntime == 0:
            ntime = 0.0000000000000001
        elif ntime == 1:
            ntime = 0.9999999999999999
        if features:
            f = '\n'.join(self.features)
            return ("%s %s %s %.16f %d\n%s\n" % (self.user, self.product, self.rating, ntime, len(self.features), f))
        else:
            return ("%s %s %s %.16f" % (self.user, self.product, self.rating, ntime))


class Candidate:
    """data structure of a candidate feature-opinion pair object"""
    def __init__(self, feature, opinion, i, j, neg, raw_opinion):
        self.feature = feature
        self.opinion = opinion
        self.i = i
        self.j = j
        self.neg = neg
        self.raw_opinion = raw_opinion

    def set_similarity_score(self, similarity_score):
        self.similarity_score = similarity_score

    def set_cluster(self, label):
        self.label = label

    def set_sentiment_score(self, sentiment_score):
        self.sentiment_score = sentiment_score
        if sentiment_score > 0:
            self.ori = "positive"
        elif sentiment_score < 0:
            self.ori = "negative"
        else:
            self.ori = "neutral"


def read_input(inp):
    """read review.txt"""
    with open(inp, 'r+') as fin:
        text = fin.read()
    return text


def save_pickle(var_to_pickle, var_string):
    """save the list/dict variable into file using pickle"""
    fileObject = open(outdir+var_string, 'wb')
    pickle.dump(var_to_pickle, fileObject)
    fileObject.close()


def load_pickle(var_string):
    """load the data from the pickle file to the list/dict variable"""
    fileObject = open(outdir+var_string, 'rb')
    var_to_load = pickle.load(fileObject)
    fileObject.close()
    return var_to_load


def str_to_time(ut):
    """Return date time from unix timestamp."""
    return datetime.datetime.fromtimestamp(int(ut))

    
def process_review(record):
    global min_time, max_time, min_time_thr, max_time_thr
    time_date = str_to_time(record[3])
    if time_date < min_time_thr or time_date > max_time_thr:
        return None
    if time_date < min_time:
        min_time = time_date
    if time_date > max_time:
        max_time = time_date
    return Review(record[1], record[0], record[2], time=time_date)
    

def construct_Reviews(inp):
    """construct a review object list, updated reviews[] and context[]"""
    print("---", str(datetime.datetime.now()), "---")
    global reviews, context, full_text, corpus, min_time, max_time
    reviews = []
    full_text = []
    corpus = []
    records = re.findall(
        r"""(?mx)
        ^product/productId:\s*(.*)\s*
        product/title:.*\s*
        product/price:.*\s*
        review/userId:\s*(.*)\s*
        review/profileName:.*\s*
        review/helpfulness:.*\s*
        review/score:\s*(.*)\s*
        review/time:\s*(.*)\s*
        review/summary:\s*(.*)\s*
        review/text:\s*(.*)\s*$""",
        read_input(inp))
    for record in records:
        r = process_review(record)
        if r == None:
            continue
        reviews.append(r)
        if(record[4] != '' and record[4][-1] != '.' and record[4][-1] != '!' and record[4][-1] != '?'):
            full_text.append(record[4] + ". " + record[5])
            corpus.append(record[4] + ". " + record[5])
        else:
            full_text.append(record[4] + " " + record[5])
            corpus.append(record[4] + " " + record[5])
    time_span_days = (max_time - min_time).days
    for r in reviews:
        ntime = r.set_ntime(time_span_days)
    print_reviews()
    print("\tAll reviews loaded; printing votes to reviews.txt")


def print_reviews():
    with open(outdir+'reviews.txt', 'w+') as fout:
        fout.write('\n'.join([review.to_str(False) for review in reviews]))


def new_run():
    """if user started a new run, delete previous candidates.txt generated(which does not match the setsize in current run)"""
    try:
        os.remove(outdir+'candidates.txt')
    except:
        pass


def import_set():
    """import trainset from previous run"""
    print("---", str(datetime.datetime.now()), "---")
    global setsize, trainset
    trainset = []
    trainset = list(int(vid) for vid in re.findall(r'\s(\d+)', read_input(outdir+'trainset.txt')))
    if(int(setsize) != len(trainset)):
        trainset = []
        print("\tInput set size does not match with previous run, generating new trainset and starting new run now")
        new_run()
        generate_set()
    else:
        print("\tTrainset imported")


def generate_set():
    """randomly select n=setsize of reviews from the entire corpus of reviews"""
    print("---", str(datetime.datetime.now()), "---")
    global trainset
    trainset = tuple(sorted(random.sample(range(0, len(reviews)), int(setsize))))
    print_trainset()
    print("\tTrainset selected; printing their ID's to trainset.txt")


def print_trainset():
    """print trainset index to trainset.txt"""
    with open(outdir+'trainset.txt', 'w+') as fout:
        fout.write('%d\n%s' % (len(trainset), ' '.join(map(str, trainset))))


def import_stopword():
    print("---", str(datetime.datetime.now()), "---")
    global stopword
    stopword_path = 'stopwords.txt'
    stopword = []
    stopword = read_input(stopword_path).split('\n')
    print('\tStopword list imported')


def data_preprocess():
    """pre-process the raw review data text"""
    print("---", str(datetime.datetime.now()), "---")
    print('\tPreprocessing data...')
    global full_text

    def preprocess(text):
        text = text.replace('!', '.')
        text = text.replace('?', '.')
        text = re.sub(r'\.+', r'.', text)
        text = re.sub(r'\.+', r'.', text)
        text = re.sub(r',+', r',', text)
        text = re.sub(r'\(+', r'(', text)
        text = re.sub(r'\)+', r')', text)
        text = re.sub(r'\.([^ ])', r'. \1', text)
        text = re.sub(r'\:([^ ])', r': \1', text)
        text = text.replace('e. g.', 'e.g.')
        text = text.replace('i. e.', 'i.e.')
        text = re.sub(r'\((.*?)\)', '', text)
        neg_mapping = [('wasnt', "wasn't"), ('werent', "weren't"), ('isnt', "isn't"), ('arent', "aren't"), ('aint', "ain't"),
                       ('havent', "haven't"), ('hasnt', "hasn't"), ('dont', "don't"), ('doesnt', "doesn't"), ('didnt', "didn't"),
                       ('wont', "won't"), ('couldnt', "couldn't"), ('wouldnt', "wouldn't"),
                       ('its', "it's"), ('thats', "that's"), ('thatre', "that're"), ('theres', "there's"), ('theyre', "they're"),
                       ('therere', "there're"), ('im', "i'm"),
                       ('Its', "It's"), ('Thats', "That's"), ('Thatre', "That're"), ('Theres', "There's"), ('Theyre', "They're"),
                       ('Therere', "There're"), ('Im', "I'm")]
        for k, v in neg_mapping:
            text = text.replace(' ' + k + ' ', ' ' + v + ' ')
        text = re.sub(r'[#|*|^|_]', r' ', text)
        text = text.replace(': )', '')
        text = text.replace(': -)', '')
        text = text.replace(': (', '')
        text = text.replace(': -(', '')
        text = text.replace(': ]', '')
        text = text.replace(': -]', '')
        text = text.replace(': [', '')
        text = text.replace(': -[', '')
        text = text.replace(': >', '')
        text = text.replace(': <', '')
        text = text.replace(': ->', '')
        text = text.replace(': -<', '')
        text = text.lower()
        return text

    full_text = list(map(preprocess, full_text))
    print("\tFinished data pre-processing")


def tokenize_corpus():
    """tokenize review text, e.g. 'this phone is good. the sound is good' -> corpus = [['this phone is good', ...],[],...]"""
    print("---", str(datetime.datetime.now()), "---")
    print('\tTokenizing reviews corpus...')
    global full_text, corpus, trainset, word_corpus
    for i, review in enumerate(full_text):
        full_text[i] = sent_tokenize(review)
    temp_word_corpus = [[[word for word in word_tokenize(sent) if word not in string.punctuation] for sent in review] for review in full_text]  # [[["It", "is", "good"],[review0sent1], ...], [....], ...]
    word_corpus = []
    for review in temp_word_corpus:
        for sent in review:
            word_corpus.append(sent)
    temp_corpus = []
    for s in trainset:
        temp_corpus.append(full_text[s])
    corpus = temp_corpus
    print_corpus()
    print("\tFinished tokenizing reviews into sentences; printing tokenized corpus to corpus.txt")


def print_corpus():
    """print tokenized corpus to corpus.txt"""
    with open(outdir+'corpus.txt', 'w+') as fout:
        fout.write('\n\n'.join([reviews[set].to_str(False) + ' ' + str(trainset.index(reviews.index(reviews[set]))) + '\n' + '\n'.join(map(str, corpus[i])) for i, set in enumerate(trainset)]))


def append_candidates(feature, opinion, i, j, neg, raw_opinion, relation, import_stage):
    """helper function to append feature pairs to candidates{} and check if there is feature already exists"""
    global candidates, features, candidates_by_review, model
    # lemmatizing feature
    """
    if(len(feature.split(' ')) > 1):
        feature = (' ').join([wnl.lemmatize(phrase, pos='n') for phrase in feature.split(' ')])
    else:
        feature = wnl.lemmatize(feature, pos='n')
    """
    features.append(feature)
    candidateObject = Candidate(feature, opinion, i, j, neg, raw_opinion)
    candidates.append(candidateObject)
    candidates_by_review[i].append(candidateObject)
    if import_stage is True:
        return
    nodes = G.nodes()
    G.add_node(feature, key='aspect')
    G.add_node(raw_opinion, key='opinion')
    edge_data = G.get_edge_data(feature, raw_opinion)
    if edge_data is None:
        G.add_edge(feature, raw_opinion, key=relation, weight=1)
    else:
        edge_relation = [i for i in edge_data]
        if relation not in edge_relation:
            G.add_edge(feature, raw_opinion, key=relation, weight=1)
        else:
            weight = edge_data[relation]['weight']
            G.add_edge(feature, raw_opinion, key=relation, weight=weight+1)
    if feature not in nodes:
        for node in nodes:
            if G.node[node]['key'] == 'aspect':
                G.add_edge(node, feature, key='aa', weight=cal_similarity_score(node, feature))
    if raw_opinion not in nodes:
        for node in nodes:
            if G.node[node]['key'] == 'opinion':
                G.add_edge(node, feature, key='oo', weight=cal_similarity_score(node, raw_opinion))


def import_candidates():
    """import previous candidates.txt"""
    print("---", str(datetime.datetime.now()), "---")
    print('\tImporting candidates from previous run...')
    global candidates, features, candidates_by_review, G
    candidates = []
    features = []
    candidates_by_review = {}
    for i in range(int(setsize)):
        candidates_by_review[i] = []
    candidates_row = read_input(outdir+'candidates.txt').split('\n')
    for pair in candidates_row:
        feature = re.search(r'^(.+):', pair).group(1)
        opinion = re.search(r': (.+) \(', pair).group(1)
        raw_opinion = opinion.split(' ')[-1]
        i = int(re.search(r'\((\d+),', pair).group(1))
        j = int(re.search(r' (\d+)\)', pair).group(1))
        neg = int(re.search(r'\)\s(.+)$', pair).group(1))
        append_candidates(feature, opinion, i, j, neg, raw_opinion, '', True)
    G = load_pickle('Graph')
    print("\tCandidates imported")


def find_adjacent_words(feature, opinion, feature_i, opinion_i, relation_list, dep):
    global model
    # ==========================
    # find feature compound
    # ==========================
    # 1. check if the current feature contains amod
    # e.g. current feature is quality -> if it is has amod of sound -> then the feature compound should be sound quality
    # ==========================
    # 2. using compound e.g. the battery life is ...
    # ==========================
    feature_compound = ''
    amod = feature
    # handle amod
    for i, relation in enumerate(reversed(relation_list)):
        reversed_i = -(i+1)
        if (relation == 'amod' and dep[reversed_i]['dependent']-1 == opinion_i):
            break
        if (relation == 'amod' and dep[reversed_i]['governorGloss'] == feature and dep[reversed_i]['governor']-1 == feature_i and 'RRB' not in dep[reversed_i]['dependentGloss']
           and 'LRB' not in dep[reversed_i]['dependentGloss'] and len(dep[reversed_i]['dependentGloss']) > 1):
            try:
                if model[dep[reversed_i]['dependentGloss'] + '_' + amod] is not None:
                    feature_compound += dep[reversed_i]['dependentGloss'] + '_'
                    amod = dep[reversed_i]['dependentGloss'] + '_' + amod
            except:
                continue
    # handle feature compound
    for i, relation in enumerate(relation_list):
        if (relation == 'compound' and dep[i]['governorGloss'] == feature and dep[i]['governor']-1 == feature_i and 'RRB' not in dep[i]['dependentGloss'] and 'LRB' not in dep[i]['dependentGloss']
           and len(dep[i]['dependentGloss']) > 1):
            feature_compound += dep[i]['dependentGloss'] + '_'
    # ==========================
    # find opinion modifier
    # ==========================
    # using advmod or amod e.g. phone is funny good / this phone is just plain stupid
    # ==========================
    opinion_modifier = ''
    for i, relation in enumerate(relation_list):
        if ((relation == 'amod' or relation == 'advmod') and dep[i]['governorGloss'] == opinion and dep[i]['governor']-1 == opinion_i and 'RRB' not in dep[i]['dependentGloss']
           and 'LRB' not in dep[i]['dependentGloss'] and len(dep[i]['dependentGloss']) > 1):
            modifier = dep[i]['dependentGloss']
            modifier_i = dep[i]['dependent']
            opinion_modifier += dep[i]['dependentGloss'] + ' '
            for j, relation in enumerate(relation_list):
                if ((relation == 'advmod' or relation == 'amod') and dep[j]['governorGloss'] == modifier and dep[j]['governor']-1 == modifier_i and 'RRB' not in dep[i]['dependentGloss']
                   and 'RRB' not in dep[i]['dependentGloss'] and len(dep[i]['dependentGloss']) > 1):
                    opinion_modifier += dep[j]['dependentGloss'] + ' '
    # ==========================
    # find negation of opinion
    # ==========================
    # 1. neg with opinion e.g. this is a not bad phone
    # 2. neg with feature e.g. this is not a bad phone
    # ==========================
    neg = ''
    for i, relation in enumerate(relation_list):
        if relation == 'neg' and ((dep[i]['governorGloss'] == opinion and dep[i]['governor']-1 == opinion_i) or (dep[i]['governorGloss'] == feature and dep[i]['governor']-1 == feature_i)):
            neg = 'not '
            break
    return (feature_compound + feature, neg + opinion_modifier + opinion, opinion)      # return like (battery life, not so good)


def extract_features_opinions():
    """find candidates by dependency parsing, which is our feature opinion pairs"""
    print("---", str(datetime.datetime.now()), "---")
    print("\tStart parsing reviews...")
    global stopword, candidates, candidates_by_review, corpus, word_corpus, features, error, model, G
    adj = ['JJ']
    noun = ['NN', 'NNS']
    verb = ['VBZ', 'VBD', 'VBP', 'VBG']
    candidates = []
    candidates_by_review = {}
    for i in range(int(setsize)):
        candidates_by_review[i] = []
    features = []
    flag = 0
    for i, review in enumerate(corpus):
        if flag % 10 == 0:
            print("\tParsed %d/%d reviews" % (flag, len(corpus)))
        flag += 1
        for j, sent in enumerate(review):
            if sent[-1] == '!' or sent[-1] == '.' or sent[-1] == '?':
                pass
            try:
                # 'tokenize,depparse,parse'
                output = nlp.annotate(sent, properties={
                  'annotators': 'tokenize, ssplit, parse, lemma, ner, dcoref',
                  'outputFormat': 'json'
                  })
                res = output['sentences'][0]['enhancedPlusPlusDependencies']
                token = output['sentences'][0]['tokens']
            except:
                # print("\tParsing error in %d, %d" % (i, j))
                error += 1
                with open(outdir+'error.txt', 'a') as fout:
                    fout.write('Parsing error in (%d, %d)\n' % (i, j))
                continue
            relation_list = []
            relation_list = [relation_dict['dep'] for relation_dict in res]
            for d_tuple in res:
                try:
                    t = ((d_tuple['governorGloss'], token[d_tuple['governor']-1]['pos'], d_tuple['governor']-1), d_tuple['dep'],
                         (d_tuple['dependentGloss'], token[d_tuple['dependent']-1]['pos'], d_tuple['dependent']-1))
                except:
                    print("\tcan't find governor/dependent in (%d, %d)" % (i, j))
                dependency = t[1]
                governor = t[0][0]
                governor_pos = t[0][1]
                governor_i = t[0][2]
                dependent = t[2][0]
                dependent_pos = t[2][1]
                dependent_i = t[2][2]
                # ------------------------------------------------------------
                # handle nsubj e.g. this phone is good
                # ------------------------------------------------------------
                if(dependency == 'nsubj'):
                    if(governor_pos in adj and dependent_pos in noun and len(dependent) > 1 and 'RRB' not in dependent and 'LRB' not in dependent and dependent not in stopword and governor not in stopword):
                        dependent, governor, raw_opinion = find_adjacent_words(dependent, governor, dependent_i, governor_i, relation_list, res)
                        append_candidates(dependent, governor, i, j, int(governor.startswith('not')), raw_opinion, 'nsubj', False)
                # ------------------------------------------------------------
                # handle amod e.g. it is a good phone.
                # ------------------------------------------------------------
                if(dependency == 'amod'):
                    if(governor_pos in noun and dependent_pos in adj and len(governor) > 1 and 'RRB' not in governor and 'LRB' not in governor and dependent not in stopword and governor not in stopword):
                        governor, dependent, raw_opinion = find_adjacent_words(governor, dependent, governor_i, dependent_i, relation_list, res)
                        try:
                            model[dependent + '_' + governor]
                        except:
                            append_candidates(governor, dependent, i, j, int(dependent.startswith('not')), raw_opinion, 'amod', False)
                # ------------------------------------------------------------
                # handle acl:relcl e.g. the phone which is good
                # ------------------------------------------------------------
                if(dependency == 'acl:relcl'):
                    if(governor_pos in noun and dependent_pos in adj and len(governor) > 1 and 'RRB' not in governor and 'LRB' not in governor and dependent not in stopword and governor not in stopword):
                        governor, dependent, raw_opinion = find_adjacent_words(governor, dependent, governor_i, dependent_i, relation_list, res)
                        append_candidates(governor, dependent, i, j, int(dependent.startswith('not')), raw_opinion, 'acl:relcl', False)
                # ------------------------------------------------------------
                # handle xcomp e.g. this phone works great!
                # ------------------------------------------------------------
                if(dependency == 'nsubj'):
                    if(governor_pos in verb and dependent_pos in noun and len(dependent) > 1 and 'RRB' not in dependent and 'RRB' not in dependent):
                        for k, relation in enumerate(relation_list):
                            if (relation == 'xcomp' and res[k]['governorGloss'] == governor and res[k]['governor']-1 == governor_i and token[res[k]['dependent']-1]['pos'] in adj):
                                governor = res[k]['dependentGloss']
                                governor_i = res[k]['dependent'] - 1
                                if (dependent not in stopword and governor not in stopword):
                                    dependent, governor, raw_opinion = find_adjacent_words(dependent, governor, dependent_i, governor_i, relation_list, res)
                                    # TODO: neg problem the neg relation is built with 'look' and 'not', not the opinion itself e.g. not looking good
                                    append_candidates(dependent, governor, i, j, int(governor.startswith('not')), raw_opinion, 'xcomp', False)
                # ------------------------------------------------------------
                # handle nsubjpass e.g. this phone is so busted!   (past tense verb as opinion)
                # ------------------------------------------------------------
                """
                if(dependency == 'nsubjpass'):
                    if(governor_pos == 'VBN' and dependent_pos in noun and len(dependent) > 1 and 'RRB' not in dependent and 'RRB' not in dependent and dependent not in stopword and governor not in stopword):
                        dependent, governor, raw_opinion = find_adjacent_words(dependent, governor, dependent_i, governor_i, relation_list, res)
                        append_candidates(dependent, governor, i, j, neg=int(governor.startswith('not')), raw_opinion, 'nsubjpass', False)
                """
                # ----------------------------------------------------------------------------------------------
                # handle dobj e.g. it is nice to have this phone/ it is nice having this phone -> dobj + xcomp
                #             e.g. it is nice that i bought this phone -> dobj + ccomp
                # ----------------------------------------------------------------------------------------------
                if(dependency == 'dobj'):
                    if(governor_pos in verb and dependent_pos in noun and len(dependent) > 1 and 'RRB' not in dependent and 'RRB' not in dependent):
                        for m, relation in enumerate(relation_list):
                            if((relation == 'xcomp' or relation == 'ccomp') and res[m]['dependentGloss'] == governor and res[m]['dependent']-1 == governor_i
                               and token[res[m]['governor']-1]['pos'] in adj):
                                governor = res[m]['governorGloss']
                                governor_i = res[m]['governor'] - 1
                                if (dependent not in stopword and governor not in stopword):
                                    dependent, governor, raw_opinion = find_adjacent_words(dependent, governor, dependent_i, governor_i, relation_list, res)
                                    append_candidates(dependent, governor, i, j, int(governor.startswith('not')), raw_opinion, 'dobj', False)
            # ----------------------------------------------------------------------------------------------------
            # handle sentence that only contains opinion e.g. Very good! -> recognize the feature as 'product'
            # ----------------------------------------------------------------------------------------------------
            if(any(x in relation_list for x in ['nsubj', 'amod', 'acl:relcl', 'nsubjpass']) is False and len(token) < 5):
                dependent = 'product'
                dependent_i = -1
                for tokens in reversed(token):
                    if tokens['pos'] == 'JJ':
                        governor = tokens['word']
                        governor_i = tokens['index']
                        dependent, governor, raw_opinion = find_adjacent_words(dependent, governor, dependent_i, governor_i, relation_list, res)
                        append_candidates(dependent, governor, i, j, int(governor.startswith('not')), raw_opinion, 'dep', False)
                        break
    save_pickle(G, 'Graph')
    print_candidates()
    print_candidates_by_review()
    print_features()
    print("\tParsed %d/%d reviews" % (len(corpus), len(corpus)))
    print("\tFinished dependency parsing with %d error(s) ; printing candidate feature opinion pairs" % (error))


def print_candidates():
    """print candidates as generated from dependency parsing"""
    global candidates
    with open(outdir+'candidates.txt', 'w+') as fout:
        fout.write('\n'.join(['%s: %s (%d, %d) %d' % (candidate.feature, candidate.opinion, candidate.i, candidate.j, candidate.neg) for candidate in candidates]))


def print_candidates_by_review():
    """print candidates_by_review{} to candidates_by_review.txt"""
    global candidates_by_review
    with open(outdir+'candidates_by_review.txt', 'w+') as fout:
        fout.write(('\n\n').join(reviews[trainset[r]].to_str(False) + '\n' + ('\n').join(['"%s %s" in sentence %d' % (candidate.opinion, candidate.feature, candidate.j)
                   for candidate in candidates_by_review[r]]) for r in candidates_by_review))


def filter_aspect_by_rank():
    print("---", str(datetime.datetime.now()), "---")
    global G, candidates, candidates_by_review, features
    print('\tPageRank algorithm...')
    pagerank = nx.pagerank_numpy(G, alpha=0.9)
    sorted_aspect = sorted(pagerank, key=pagerank.get)
    for node in reversed(sorted_aspect):
        if G.node[node]['key'] == 'opinion':
            sorted_aspect.remove(node)
    pagerank_filter = sorted_aspect[:int(len(sorted_aspect)*0.2)]
    counter = 0
    for candidate in reversed(candidates):
        if candidate.feature in pagerank_filter:
            counter += 1
            review_index = candidate.i
            if candidate in candidates_by_review[review_index]:
                candidates_by_review[review_index].remove(candidate)
            features = list(filter(lambda feature: feature != candidate.feature, features))
            candidates.remove(candidate)
    print('\t%d aspect terms and %d candidates are filtered by PageRank algorithm' % (len(pagerank_filter), counter))


def word_freq_lookup(text, wordlist):
    # global full_text, corpus, full_wordlist, corpus_wordlist
    tokenizer = RegexpTokenizer(r'\w+')
    for review in text:
        for sent in review:
            for word in tokenizer.tokenize(sent):
                if(stem(word.lower()) not in wordlist):
                    wordlist[stem(word.lower())] = 1
                else:
                    wordlist[stem(word.lower())] += 1


def construct_wordlist():
    """construct a list that store all the stemmed distinct words and their corresponding appearing location, which provides freq to us"""
    print("---", str(datetime.datetime.now()), "---")
    global full_text, corpus, full_wordlist, corpus_wordlist
    full_wordlist = {}          # full_wordlist{} = {'it': n, 'is': n} where n is the freq of the word
    corpus_wordlist = {}
    word_freq_lookup(full_text, full_wordlist)
    word_freq_lookup(corpus, corpus_wordlist)
    print_full_wordlist()
    print_corpus_wordlist()
    print("\tConstructed wordlist; printing wordlist to full_wordlist.txt and corpus_wordlist.txt")


def print_full_wordlist():
    """print full_wordlist to full_wordlist.txt"""
    with open(outdir+'full_wordlist.txt', 'w+') as fout:
            fout.write('\n'.join(['%s: %d' % (word, full_wordlist[word]) for word in full_wordlist]))


def print_corpus_wordlist():
    """print corpus_wordlist to corpus_wordlist.txt"""
    with open(outdir+'corpus_wordlist.txt', 'w+') as fout:
            fout.write('\n'.join(['%s: %d' % (word, corpus_wordlist[word]) for word in corpus_wordlist]))


def construct_features():
    """generate sorted list of distinct lemmatized features according to candidates{}"""
    print("---", str(datetime.datetime.now()), "---")
    print('\tGenerating features from candidates...')
    global features, candidates
    features = []
    for f in candidates.keys():
        appearences = len(candidates[f][0])
        features += appearences*[f]
    features = sorted(features)
    print_features()
    print('\tGenerated final list of features; printing sorted features list to features.txt')


def print_features():
    """print features[] to features.txt"""
    global features
    unique_features = set(features)
    with open(outdir+'features.txt', 'w+') as fout:
        fout.write('Number of total features: %s\nNumber of distinct features: %s' % (str(len(features)), str(len(unique_features))) + '\n\n' + '\n'.join(sorted(features)))


def build_word2vec():
    """build word2vec model from word_corpus[]"""
    print("---", str(datetime.datetime.now()), "---")
    global word_corpus, model, bigram, trigram
    print('\tBuilding word2vec language model...')
    if (os.path.isfile(outdir+'word2vec.model') and os.path.isfile(outdir+'bigram.model') and os.path.isfile(outdir+'trigram.model')):
        bigram = gensim.models.Phrases.load(outdir+'bigram.model')
        trigram = gensim.models.Phrases.load(outdir+'trigram.model')
        model = gensim.models.Word2Vec.load(outdir+'word2vec.model')
        print('\tSuccessfully loaded previous trained word2vec model')
    else:
        bigram = gensim.models.Phrases(word_corpus)
        trigram = gensim.models.Phrases(bigram[word_corpus])
        model = gensim.models.Word2Vec(trigram[word_corpus], size=300, alpha=0.025, window=5, min_count=1, max_vocab_size=None, sample=0.001, seed=1, workers=3)
        bigram.save(outdir+'bigram.model')
        trigram.save(outdir+'trigram.model')
        model.save(outdir+'word2vec.model')
        print('\tSuccessfully built and trained word2vec model')


def import_similarity_score():
    """import similarity score from similarity_score.txt"""
    print("---", str(datetime.datetime.now()), "---")
    print('\tLoading similarity score array pickle from previous run...')
    global score_array
    score_array = load_pickle('score_array')
    score_array = score_array.astype(np.float64)
    print("\tSimilarity scores array imported")


def cal_similarity_score(word1, word2):
    global model
    """calculate similarity score of two words"""
    # if(word1 == word2):
    # return 12876699.5                   # as calculated from http://ws4jdemo.appspot.com/
    # semcor_ic = wordnet_ic.ic('ic-semcor.dat')
    """
    highest_score = 0
    word1 = wn.synsets(word1, pos=wn.NOUN)
    word2 = wn.synsets(word2, pos=wn.NOUN)
    for synset1 in word1:
        for synset2 in word2:
            # score = synset1.jcn_similarity(synset2, semcor_ic)
            score = synset1.lch_similarity(synset2)
            if(score > highest_score):
                highest_score = score
    return highest_score
    """
    wword1 = word1
    wword2 = word2
    while True:
        word1_valid = False
        word2_valid = False
        try:
            if model[word1] is not None:
                word1_valid = True
        except:
            word2vec_error_log(word1)
        try:
            if model[word2] is not None:
                word2_valid = True
        except:
            word2vec_error_log(word2)
        if word1_valid is True and word2_valid is True:
            break
        if word1 == '':
            word2vec_error_log(wword1)
            break
        if word2 == '':
            word2vec_error_log(wword2)
            break
        if word1_valid is False:
            word1 = '_'.join(word1.split('_')[1:])
        if word2_valid is False:
            word2 = '_'.join(word2.split('_')[1:])
    """
    try:
        if model[word1] is not None:
            pass
        else:
            raise KeyError('no %s in model', word1)
        if model[word2] is not None:
            pass
        else:
            raise KeyError('no %s in model', word2)
    except KeyError as e:
        print('\tword2vec error ', e)
        exit()
    """
    if word1 != '' and word2 != '':
        return model.similarity(word1, word2)
    else:
        return 0


def word2vec_error_log(error_word):
    """print error log to word2veclog.txt when there is error occured in word2vec similarity calculation"""
    """
    with open(outdir+'word2veclog.txt', 'a') as fout:
        fout.write('%s cannot be found in word2vec model\n' % (error_word))
    """
    pass


def construct_similarity_score():
    """calculate Jcn score for each feature pairs"""
    print("---", str(datetime.datetime.now()), "---")
    print('\tConstructing similarity score array...')
    global features, candidates, score_array, temp_score_array
    temp_score_array = []
    """
    for i, feature1 in enumerate(features):
        temp_score_array.append([])
        for j, feature2 in enumerate(features):
            score = cal_similarity_score(feature1, feature2)
            temp_score_array[-1].append(score)
        candidates[i].set_similarity_score(temp_score_array[-1])
    """
    for feature in features:
        while True:
            try:
                if feature == '':
                    temp_score_array.append([0]*300)
                    break
                if model[feature] is not None:
                    temp_score_array.append(model[feature])
                    break
            except:
                feature = '_'.join(feature.split('_')[1:])
    score_array = np.array(temp_score_array, dtype=np.float64)
    save_pickle(score_array, 'score_array')
    print('\tConstructed similarity score for each feature; pickling score matrix to score_array')


def print_score_array():
    """print score_dict to similarity_score.txt"""
    global features, score_dict, temp_score_array, score_array
    with open(outdir+'similarity_score.txt', 'w+') as fout:
        fout.write('\n'.join(['%s: [' % (feature) + ', '.join(['%.6f' % (score) for score in temp_score_array[i]]) + ']' for i, feature in enumerate(features)]))


def import_clusters():
    """import rought clusters to clusters[]"""
    print("---", str(datetime.datetime.now()), "---")
    print('\tImporting clusters from previous run...')
    global candidates, clusters, clusters_dict
    clusters_rows = read_input(outdir+'clusters.txt').split('\n')
    clusters = []
    clusters_dict = {}
    for row in clusters_rows:
        if row != '':
            clusters.append(float(row))
    for i, labels in enumerate(clusters):
        if(labels in clusters_dict.keys()):
            clusters_dict[labels].append(candidates[i])
        else:
            clusters_dict[labels] = [candidates[i]]
    print('\tClusters imported')


def plot_clustering(X_red, X, labels, k, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        if list(labels).count(labels[i]) > 20:
            plt.scatter(X_red[i, 0], X_red[i, 1], s=8, marker='o', color=plt.cm.spectral(labels[i]/float(k)))
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    fig.savefig(outdir+'clustering.png')


def cluster_features():
    """cluster final candidates into groups"""
    print("---", str(datetime.datetime.now()), "---")
    global features, candidates, score_array, clusters, clusters_dict
    k = int(len(features)*0.3)
    print('\tClustering features with %s clusters...' % (str(k)))
    clustering = KMeans(n_clusters=k, random_state=0).fit(score_array)
    # clustering = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='ward').fit(score_array)
    # clustering = SpectralClustering(n_clusters=k).fit(score_array)
    clusters = clustering.labels_
    # ===================
    # k- medoids
    """
    M, C = kmedoids.kMedoids(score_array, k)
    clusters = [None] * len(features)
    for i, label in enumerate(C):
        if C[label] == []:
            continue
        for j, array_idx in enumerate(C[label]):
            clusters[array_idx] = label
    """
    # ===================
    print('\tClustered features; embedding feature matrix for visualization')
    embedded_score_array = PCA(n_components=2).fit_transform(score_array)
    print('\tDrawing clustering result to clustering.png...')
    plot_clustering(embedded_score_array, score_array, clusters, k, 'Clustering of features')
    clusters_dict = {}
    for i, labels in enumerate(clusters):
        candidates[i].set_cluster(labels)
        if(labels in clusters_dict.keys()):
            clusters_dict[labels].append(candidates[i])
        else:
            clusters_dict[labels] = [candidates[i]]
    print('\tFinished clustering; printing clusters to clusters.txt')
    print_clusters()


def print_clusters():
    """print clusters to clusters.txt and print clusters_dict to clusters_dict.txt"""
    global clusters, clusters_dict
    np.savetxt(outdir+'clusters.txt', clusters, delimiter=', ')
    with open(outdir+'clusters_dict.txt', 'w+') as fout:
        output = 'Number of clusters: %s\n\n' % (len(set(clusters)))
        for label in set(clusters):
            sorted_features_for_output = sorted([candidate.feature for candidate in clusters_dict[label]])
            output += 'cluster %d:\n' % (label) + '\n'.join(sorted_features_for_output) + '\n\n'
        fout.write(output)


def cal_sentiment_score_of_single_word(word):
    """calculate sentiment score of a single word only"""
    synset = wn.synsets(word)
    score = 0
    weight = 0
    flag = 0
    if synset != []:
        for syn in synset:
            if (syn.pos() == 'n' or syn.pos() == 'v'):
                continue
            senti = swn.senti_synset(syn.name())
            pos = senti.pos_score()
            neg = senti.neg_score()
            score += (pos-neg)/(flag+1)
            weight += 1/(flag+1)
            flag += 1
        if weight == 0:
            return 0
        else:
            return score/weight
    else:
        return 0


def cal_sentiment_score_of_opinion_phrase(opinion):
    """calculate sentiment score of an opinion phrase"""
    if opinion.startswith('not '):
        neg = True
        opinion = [spell(word) for word in opinion.split(' ')][1:]
    else:
        neg = False
        opinion = [spell(word) for word in opinion.split(' ')]
    scores = [cal_sentiment_score_of_single_word(word) for word in opinion]

    def inner_senti_score(scores):
        if len(scores) == 1:
            return scores[0]
        new_scores = []
        if scores[-1] > 0 and scores[-2] < 0:
            res = scores[-1] - (scores[-1]*abs(scores[-2]))
            new_scores = scores[:-2] + [res]
        else:
            if scores[-1] > 0:
                sigma = 1
            elif scores[-1] == 0:
                sigma = 0
            else:
                sigma = -1
            res = sigma*(abs(scores[-1]) + (1 - abs(scores[-1]))*scores[-2])
            new_scores = scores[:-2] + [res]
        return inner_senti_score(new_scores)
    neg = -1 if neg else 1
    return neg*inner_senti_score(scores)


def sentiment_scoring():
    """give sentiment score to each feature opinion pair"""
    print("---", str(datetime.datetime.now()), "---")
    print('\tCalculating sentiment score of each candidates...')
    global clusters_dict, clusters_for_candidates, filtered_candidates_key
    for candidate in candidates:
        candidate.set_sentiment_score(cal_sentiment_score_of_opinion_phrase(candidate.opinion))
    clusters_for_candidates = clusters_dict
    filtered_candidates_key = []
    for label in clusters_for_candidates:
        clusters_for_candidates[label].sort(key=lambda x: x.sentiment_score)
        lower = math.floor(len(clusters_for_candidates[label]) * 0.4)
        upper = math.ceil(len(clusters_for_candidates[label]) * (1 - 0.4))
        filtered_candidates_key += clusters_for_candidates[label][lower:upper]
    print('\tCalculated sentiment score of candidates; printing clusters for candidates (which sort candidates in each cluster by their score) to clusters_for_candidates.txt')
    print_clusters_for_candidates()


def print_clusters_for_candidates():
    """print clusters_for_candidates to clusters_for_candidates.txt"""
    global clusters, clusters_for_candidates
    with open(outdir+'clusters_for_candidates.txt', 'w+') as fout:
        output = 'Number of clusters: %s\n\n' % (len(set(clusters)))
        for label in set(clusters):
            features_sorted_by_senti_score = ["['%s', %f, '%s', %d, %d]" % (candidate.feature, candidate.sentiment_score, candidate.opinion, candidate.i, candidate.j)
                                              for candidate in clusters_for_candidates[label]]
            output += 'cluster %d:\n' % (label) + '\n'.join(features_sorted_by_senti_score) + '\n\n'
        fout.write(output)


def filter_candidates_by_clusters():
    """filter those candidates in every cluster that are not the top n or low n sentiment score"""
    print("---", str(datetime.datetime.now()), "---")
    print('\tFiltering candidates by their sentiment score in their cluster...')
    global candidates, filtered_candidates_key, final_candidates, candidates_by_review
    final_candidates = candidates
    for candidate_to_filter in filtered_candidates_key:
        if candidate_to_filter in final_candidates:
            final_candidates.remove(candidate_to_filter)
        review_index = candidate_to_filter.i
        if candidate_to_filter in candidates_by_review[int(review_index)]:
            candidates_by_review[review_index].remove(candidate_to_filter)
    print('\tFiltered candidates by clusters; printing final candidates to final_candidates.txt')
    print_final_candidates()


def print_final_candidates():
    """print candidates as generated from dependency parsing"""
    global final_candidates
    with open(outdir+'final_candidates.txt', 'w+') as fout:
        fout.write('\n'.join(['%s: %s (%d, %d) %d' % (candidate.feature, candidate.opinion, candidate.i, candidate.j, candidate.neg) for candidate in candidates]))


def output_sentiment():
    """final step of the system, output a pretty formatted summary"""
    print("---", str(datetime.datetime.now()), "---")
    print('\tGenerating final summary; printing final summary to summary.txt')
    print_output_sentiment()
    print_summary()


def print_summary():
    global candidates_by_review
    with open(outdir+'summary.txt', 'w+') as fout:
        fout.write(('\n\n').join(reviews[trainset[r]].to_str(False) + '\n' + ('\n').join(['%s, (%s, %s), %d' % (candidate.feature, candidate.opinion, candidate.ori, candidate.neg)
                   for candidate in candidates_by_review[r]]) for r in candidates_by_review))


def print_output_sentiment():
    global candidates_by_review
    with open(outdir+'output_sentiment.txt', 'w+') as fout:
        fout.write(('\n\n').join(reviews[trainset[r]].to_str(False) + ' ' + str(len(candidates_by_review[r])) + '\n' + ('\n').join(['%s, (%s, %s), %d' % (' '.join(candidate.feature.split('_')), candidate.raw_opinion, candidate.ori, candidate.neg)
                   for candidate in candidates_by_review[r]]) for r in candidates_by_review))


def main():
    global outdir, source
    existfile = os.path.isfile
    if existfile(source):
        construct_Reviews(source)
    else:
        print("\tNo such file %s found" % (source))
    if(int(setsize) > len(reviews)):
        print("\tWrong input: please input a valid number of review you want to analyze")
        exit()
    # -------------------------------------------------------------------------------------
    import_set() if existfile(outdir+'trainset.txt') else generate_set()
    import_stopword()
    data_preprocess()
    tokenize_corpus()
    build_word2vec()
    import_candidates() if existfile(outdir+'candidates.txt') else extract_features_opinions()
    filter_aspect_by_rank()
    # construct_wordlist()
    # construct_features()
    # -------------------------------------------------------------------------------------
    import_similarity_score() if existfile(outdir+'score_array') else construct_similarity_score()
    import_clusters() if existfile(outdir+'clusters.txt') else cluster_features()
    sentiment_scoring()
    filter_candidates_by_clusters()
    output_sentiment()
    # -------------------------------------------------------------------------------------
    print("---", str(datetime.datetime.now()), "---")
    print("\tSentiment analysis done")
    print("---", str(datetime.datetime.now()), "---")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("\tWrong input: please input as follows: python feature_extraction.py [source] [outout] [setsize]")
        exit()
    source = sys.argv[1]
    outdir = sys.argv[2]
    setsize = sys.argv[3]
    main()
