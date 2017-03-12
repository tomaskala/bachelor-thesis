import logging
import os
from time import time

import gensim

GENSIM_OUT = './gensim'
WORD2VEC_PATH = os.path.join(GENSIM_OUT, './word2vec_lemma_skipgram')

# Manually taken from the dataset. Some words are malformed due to stemming or lemmatization.
CZECH_STOPWORDS = frozenset(
    ['282', '45924', '468', 'adalší', 'adblock', 'adsbygoogl', 'adsbygoogle', 'advide', 'aftershar', 'aftershare',
     'allows', 'although', 'alway', 'anchory', 'api', 'apod', 'apod.step', 'appenda', 'appendchild', 'arrayd',
     'arrayde', 'async', 'bbcod', 'bbcode', 'befor', 'before', 'blesk.cz', 'blesk.cz.auto', 'btn', 'callback',
     'choosen', 'clankyodjinud', 'clankyvideoportal', 'click', 'com', 'comment', 'comments', 'config', 'configuration',
     'content', 'controls', 'copypast', 'copyright', 'count', 'createelement', 'cs', 'css', 'css2',
     'd.body.appendchild', 'd.createelement', 'd.getelementbyid', 'd.getelementsbytagname', 'd.location', 'default',
     'defaults', 'description', 'diskuse.pleas', 'disqus', 'disqus.nejnov', 'div', 'document', 'document.createelement',
     'document.getelementsbytagname', 'document.titel', 'document.write', 'donation', 'dotazy.step', 'dropdown', 'dsq',
     'dsq.typat', 'dsq.type', 'dynamiccallback', 'echo24cz', 'edit', 'edita', 'editbety', 'elm', 'embedcode', 'enabl',
     'enable', 'enabled', 'escap', 'escape', 'exampl', 'example', 'facebook.com', 'fals', 'false', 'fb', 'fbs',
     'fbs_click', 'fjs', 'fjs.parentnode.insertbefore', 'follow', 'format', 'formatb', 'formatba', 'formatovan',
     'formatovani', 'formatovat', 'formhtml', 'forum', 'forum.valka.cz', 'from', 'function', 'functions',
     'functions.najet', 'functions.pole', 'functions.zelna', 'galid', 'gallerycarousel', 'galurl', 'gatrackevent',
     'gatrackevents', 'gatracksocialinteractions', 'gcfg', 'getelementbyid', 'getelementsbytagnam', 'getjson',
     'getpocket', 'getstats', 'googlefillslot', 'hawaj.cz', 'head', 'height', 'href', 'html', 'http', 'https',
     'https://widgets.getpocket.com/v1/j/btn.js?v=1', 'i18n', 'i18n.readmore', 'iconk', 'iconka', 'id', 'ida', 'idy',
     'if', 'ihned.cz', 'ihnedcz', 'inc', 'informace.step', 'initcallback', 'insertbefor', 'insertbefore', 'into',
     'isport.cz', 'itemloadcallback', 'j.id', 'j.src', 'j.srciální', 'javascript', 'jquer', 'js', 'js.id', 'js.src',
     'js.srciální', 'json', 'kliknutí', 'lang', 'left', 'link', 'links', 'local', 'locale', 'localisation', 'location',
     'location.href', 'mainl', 'mainly', 'method', 'mobileoverla', 'mobileoverlej', 'mous', 'nanoflowcell.ag',
     'navigation', 'navigation.zde', 'neplatila.step', 'networking.zde', 'null', 'onafteranimation',
     'onbeforeanimation', 'over', 'override', 'page', 'palm', 'parentnod', 'parentnode', 'pasting', 'php', 'pick',
     'platform', 'platform.twitter.com/widgets.js',
     "platform.twitter.com/widgets.js';fjs.parentnode.insertbefore(js,fjs)", 'played', 'pleas', 'plugins', 'po.src',
     'po.typat', 'po.type', 'pocket', 'pos', 'position', 'powered', 'ppc', 'ppc_', 'preddefin', 'profimedia.cz',
     'publisherke', 'publisherkey', 'push', 'pwidget', 'queu', 'queue.push', 'readmor', 'replac', 'replace', 'required',
     'requiréd', 'restserver', 'return', 'rhhar0uejt6tohi9', 'rss', 's.async', 's.parentnode.insertbefore', 's.src',
     's.type', 'sablon', 'sablona', 'sablona.step', 'sashec', 'saved', 'script', 'scriptum', 'search', 'search.pokud',
     'secondtracker', 'see', 'setup.authory', 'share.tweet.kl', 'sharepopups', 'sharequot', 'sharer', 'shortnam',
     'shortnamat', 'shortname', 'showerror', 'size', 'sja', 'sklikarticleinicialized', 'sklikd', 'sklikdat',
     'sklikdata', 'sklikde', 'sklikreklam', 'sklikreklama', 'sklikreklama_', 'sport.cz', 'src', 'stanice.port',
     'stretching', 'success', 'switch', 'tag', 'tagu', 'tel', 'text', 'the', 'this', 'tip.although', 'titel', 'titl',
     'title', 'tn.cz', 'toolbar', 'trackingobject', 'transitioncarousel', 'translate', 'translated', 'translation',
     'true', 'twitter', 'type', 'u00edc', 'ulékař', 'url', 'urls', 'user', 'using', 'uzivatel', 'valka.cz', 'var',
     'vara', 'variable', 'variables', 'view', 'vlajku.step', 'wallpaper', 'webpag', 'webpage', 'whe', 'whole', 'widget',
     'widgets', 'width', 'will', 'window', 'window.adsbygoogle', 'window.sklikarticleinicialized', 'with', 'wjs',
     'writ', 'write', 'wsj', 'www', 'xmlad', 'your', 'zoneid', 'zvyraznit', 'ácí', 'áhnout', 'ální', 'ání', 'átor',
     'áva', 'ávě', 'éra', 'ící', 'ílat', 'íčový', 'čka', 'čnost', 'čný', 'čít', 'ění', 'ětý', 'ětšit', 'řit', 'šit',
     'šný', 'ště', 'ždý', 'žský'])


class LemmaPreprocessor:
    def __init__(self, documents, include_names, min_length=3, max_length=15):
        self.documents = documents
        self.include_names = include_names
        self.min_length = min_length
        self.max_length = max_length

    def __iter__(self):
        include_names = self.include_names
        min_length = self.min_length
        max_length = self.max_length

        for doc in self.documents:
            if include_names:
                yield [word for word in doc.name if
                       word.lower() not in CZECH_STOPWORDS and min_length <= len(word) <= max_length]

            yield [word for word in doc.text if
                   word.lower() not in CZECH_STOPWORDS and min_length <= len(word) <= max_length]


def perform_word2vec(fetcher, include_names, min_length=3, max_length=15):
    t = time()

    if os.path.exists(WORD2VEC_PATH):
        logging.info('Loading Word2Vec')
        word2vec_model = gensim.models.Word2Vec.load(WORD2VEC_PATH)
        logging.info('Loaded Word2Vec in %fs.', time() - t)
    else:
        logging.info('Training Word2Vec')
        corpus = LemmaPreprocessor(fetcher, include_names, min_length, max_length)
        word2vec_model = gensim.models.Word2Vec(corpus, size=100, window=5, min_count=10, iter=5, sg=1)
        word2vec_model.save(WORD2VEC_PATH)
        logging.info('Created and saved Word2Vec in %fs.', time() - t)

    return word2vec_model
