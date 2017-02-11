import logging
from time import time


class LemmatizedDocument:
    __slots__ = ['doc_id', 'name', 'text', 'similarity']

    def __init__(self, doc_id, original_document, similarity=None):
        self.doc_id = doc_id
        self.name = original_document.name
        self.text = original_document.text
        self.similarity = similarity

    def __str__(self):
        return '{:d}: {:s}'.format(self.doc_id, ' '.join(self.name))


def docids2documents(events, fetcher):
    """
    Retrieve the actual text documents for a collection events from the document IDs produced by the `postprocessing`
    module. Do this by grouping all event documents together, retrieving them in a single pass over the corpus and then
    redistributing them back to their respective events. This is to maintain some level of efficiency, since the
    lemmatized collection takes a while to iterate over.
    :param events: list of events in the format outputted by the `postprocessing.keywords2docids` functions, either
        the ones containing only document IDs or those containing both document IDs and their similarities
    :param fetcher: document fetcher to use for document streaming, should be for the same dataset as the one used for
        event detection, but can have different settings (forms instead of lemmas, different POS tags)
    :return: list of lists with each inner list representing an event and consisting of `LemmatizedDocument` objects,
        the order of the events is preserved
    """
    t = time()
    logging.info('Retrieving documents for %d events.', len(events))
    docids = []

    # Collect document IDs for all events altogether and retrieve them at once, so the collection is iterated only once.
    for event in events:
        for _, _, burst_docs in event:
            if type(burst_docs[0]) is tuple:
                # If K-NN was used to retrieve the documents, each document is a tuple (doc_id, doc_similarity).
                docs = list(map(lambda item: item[0], burst_docs))
                docids.extend(docs)
            else:
                docids.extend(burst_docs)

    docids2docs = load_documents(docids, fetcher)
    events_out = []

    # Redistribute the documents back to the individual events, keeping similarities if they were retrieved previously.
    for event in events:
        event_out = []

        for burst_start, burst_end, burst_docs in event:
            if type(burst_docs[0]) is tuple:
                docs_out = [LemmatizedDocument(doc_id, docids2docs[doc_id], similarity) for doc_id, similarity in
                            burst_docs]
            else:
                docs_out = [LemmatizedDocument(doc_id, docids2docs[doc_id]) for doc_id in burst_docs]

            event_out.append((burst_start, burst_end, docs_out))

        events_out.append(event_out)

    logging.info('Retrieved event documents in %fs.', time() - t)
    return events_out


def load_documents(docids, fetcher):
    """
    Load the documents with the given indices from disk.
    :param docids: IDs of documents to be loaded, will be unique-d and sorted
    :param fetcher: document fetcher to use for document streaming
    :return: dictionary mapping document IDs to the retrieved documents
    """
    if len(docids) == 0:
        raise ValueError('No document IDs given.')

    docids = list(sorted(set(docids)))
    documents = []
    doc_pos = 0

    for doc_id, document in enumerate(fetcher):
        if doc_id == docids[doc_pos]:
            documents.append(document)
            doc_pos += 1

        if doc_pos == len(docids):
            break

    return dict(zip(docids, documents))


def main():
    from event_detection import data_fetchers
    events = [[(2, 3, [(0, 100), (2, 102), (3, 103)]), (4, 7, [(5, 205), (9, 209), (13, 213)])],
              [(1, 8, [(1, 301), (4, 304), (2, 302)]), (2, 3, [(0, 400), (2, 402), (3, 403)])],
              [(3, 6, [(8, 508), (5, 505), (2, 502)])]]
    new_events = docids2documents(events, data_fetchers.CzechLemmatizedTexts(dataset='dec-jan', fetch_forms=True))

    for i, event in enumerate(new_events):
        print('Event {:d}'.format(i))

        for burst_start, burst_end, burst_docs in event:
            print('Burst: ({:d}, {:d})'.format(burst_start, burst_end))

            for doc in burst_docs:
                print(doc, doc.similarity)

        print()


if __name__ == '__main__':
    main()
