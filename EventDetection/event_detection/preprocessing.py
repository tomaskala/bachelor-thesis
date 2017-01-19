import logging
import os
from time import time

import gensim
import numpy as np

from event_detection.sphere import SphericalKMeans

GENSIM_OUT = './gensim'
DOC2VEC_PATH = os.path.join(GENSIM_OUT, './doc2vec_sw')

# Manually taken from the dataset. Some words are malformed due to stemming.
CZECH_STOPWORDS = frozenset(
    ['adsbygoogl', 'adsbygoogle', 'advide', 'aftershar', 'although', 'api', 'appendchild', 'arrayd',
     'async', 'bbcod', 'befor', 'before', 'btn', 'callback', 'choosen', 'clankyodjinud',
     'clankyvideoportal', 'click', 'com', 'comments', 'config', 'configuration', 'copypast', 'count',
     'createelement', 'cs', 'css2', 'defaults', 'description', 'disqus', 'document', 'donation',
     'dropdown', 'dsq', 'dynamiccallback', 'echo24cz', 'edit', 'edita', 'elm', 'enabl', 'enabled',
     'escap', 'escape', 'exampl', 'example', 'fals', 'false', 'fbs', 'fbs_click', 'fjs', 'format', 'formatb',
     'formatovan', 'formatovat', 'formhtml', 'forum', 'function', 'functions', 'galid', 'gallerycarousel',
     'galurl', 'gatrackevents', 'gatracksocialinteractions', 'gcfg', 'getelementbyid',
     'getelementsbytagnam', 'getjson', 'getpocket', 'getstats', 'head', 'height', 'href', 'html', 'http',
     'https', 'i18n', 'iconk', 'id', 'ida', 'if', 'ihnedcz', 'initcallback', 'insertbefor',
     'insertbefore', 'into', 'itemloadcallback', 'javascript', 'js', 'json', 'lang', 'left', 'link',
     'links', 'local', 'location', 'mainl', 'method', 'mobileoverla', 'mous', 'navigation', 'null',
     'onafteranimation', 'onbeforeanimation', 'parentnod', 'parentnode', 'pasting', 'php', 'platform',
     'pleas', 'plugins', 'pocket', 'pos', 'position', 'powered', 'ppc', 'ppc_', 'preddefin',
     'publisherke', 'push', 'pwidget', 'queu', 'readmor', 'replac', 'replace', 'required', 'restserver',
     'return', 'rhhar0uejt6tohi9', 'sablon', 'sashec', 'script', 'scriptum', 'search', 'secondtracker',
     'sharepopups', 'sharequot', 'sharer', 'shortnam', 'shortname', 'showerror', 'size',
     'sklikarticleinicialized', 'sklikd', 'sklikdata', 'sklikreklam', 'sklikreklama_', 'src',
     'stretching', 'success', 'tagu', 'text', 'the', 'titl', 'title', 'toolbar', 'trackingobject',
     'transitioncarousel', 'translated', 'translation', 'true', 'type', 'u00edc', 'url', 'urls', 'var',
     'variables', 'view', 'wallpaper', 'webpag', 'webpage', 'widget', 'widgets', 'width', 'window', 'with', 'wjs',
     'writ', 'write', 'wsj', 'www', 'xmlad', 'your', 'zoneid'])


class DocumentTagger:
    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        for i, doc in enumerate(self.documents):
            tags = [i]
            words = [word for word in gensim.utils.simple_preprocess(doc.text) if word not in CZECH_STOPWORDS]
            tagged_doc = gensim.models.doc2vec.TaggedDocument(words, tags)

            yield tagged_doc


class Preprocessor:
    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        for doc in self.documents:
            yield [word for word in gensim.utils.simple_preprocess(doc.text) if word not in CZECH_STOPWORDS]


def perform_doc2vec(fetcher):
    doc_tagger = DocumentTagger(fetcher)
    t = time()

    if os.path.exists(DOC2VEC_PATH):
        logging.info('Loading Doc2Vec')
        doc2vec_model = gensim.models.Doc2Vec.load(DOC2VEC_PATH)
        logging.info('Loaded Doc2Vec in %fs.', time() - t)
    else:
        logging.info('Training Doc2Vec')
        doc2vec_model = gensim.models.Doc2Vec(doc_tagger, dm=1, dm_concat=1, size=100, negative=5, hs=0, min_count=5,
                                              window=5, iter=5)
        doc2vec_model.save(DOC2VEC_PATH)
        logging.info('Created and saved Doc2Vec in %fs.', time() - t)

    return doc2vec_model


# TODO: Try spherical K-Means.
def cluster_documents(documents, n_clusters=15):
    t = time()

    # normalize(documents, norm='l2', copy=False)
    # clusterer = MiniBatchKMeans(n_clusters=n_clusters, n_init=10, random_state=1)
    clusterer = SphericalKMeans(n_clusters=n_clusters)
    labels = clusterer.fit_predict(documents)

    logging.info('Performed clustering into %d clusters in %fs.', n_clusters, time() - t)

    clusters = [[] for _ in range(n_clusters)]

    for document_ix, label in np.ndenumerate(labels):
        clusters[label].append(document_ix[0])

    for i, c in enumerate(clusters):
        logging.info('Cluster #%d of %d documents', i, len(c))

    return clusters
