from django.shortcuts import render

from pykeen.pipeline import pipeline
from pykeen.models import TransE
from pykeen.sampling import BasicNegativeSampler
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory
from pykeen.predict import predict_target, predict_all


import pandas as pd
import numpy as np
import scipy
import json


EPOCH = 100
DEVICE = 'mps'
CONFIDENCE_THRESHOLD = .6


def handle_linkpred_req(request):
    context = {}

    triples = [
        ('Heart', 'is_a', 'student'),
        ('Chie', 'is_a', 'student'),
        ('Heart', 'is_a', 'human'),
        ('Chie', 'is_a', 'human'),
        ('Heart', 'like', 'cat'),
        ('Chie', 'like', 'cat'),
        ('Chie', 'like', 'dog'),
        ('Heart', 'like', 'pig'),
        ('Chie', 'is_a', 'girl'),
        ('Heart', 'is_a', 'boy'),
        ('Heart', 'is_a', 'god'),
    ]

    df = pd.DataFrame(triples, columns=('subject', 'predicate', 'object'))

    recontructed_graph = link_prediction(df)
    context['recontructed_graph'] = recontructed_graph

    # recontructed_graph_df = pd.DataFrame(recontructed_graph, columns=('subject', 'predicate', 'object', 'group'))
    # context['graph'] = graph2json(recontructed_graph_df)

    df['group'] = 0
    context['graph'] = graph2json(df)

    return render(request, 'LinkPrediction/link_prediction.html', context)


def graph2json(df):
    graph = {'nodes': [], 'links': []}

    unique = np.unique(df[['subject', 'object']].values)

    for val in unique:
        graph['nodes'].append({'id': val})

    for _, row in df.iterrows():
        graph['links'].append({
            'source': row['subject'],
            'relation': row['predicate'],
            'target': row['object'],
            'group': row['group'],
        })
    
    return json.dumps(graph)


def link_prediction(df):
    tf = TriplesFactory.from_labeled_triples(
      triples = df[['subject', 'predicate', 'object']].values,
    )
    
    result = pipeline(
        training = tf,
        testing = tf,
        model = TransE,
        device = DEVICE,
        optimizer = 'Adam',
        training_loop = 'sLCWA',
        epochs = EPOCH,
        negative_sampler = BasicNegativeSampler,
        evaluator = RankBasedEvaluator,
        random_seed = 1729,
    )

    ###
    # id2ent = tf.entity_id_to_label
    # id2rel = tf.relation_id_to_label
    # recontructed_graph = []
    # max_link = len(df) * 2
    # pred = predict_all(model=result.model)
    # for each_triple in pred.result:
    #     subject = id2ent[int(each_triple[0])]
    #     predicate = id2rel[int(each_triple[1])]
    #     object = id2ent[int(each_triple[2])]
    #     if subject != object:
    #         recontructed_graph.append((subject, predicate, object, 'predict'))
    #         if len(recontructed_graph) >= max_link:
    #             break
    
    # for _, row in df.iterrows():
    #     gr_triple = tuple(row)
    #     if gr_triple in recontructed_graph:
    #         recontructed_graph[recontructed_graph.find(gr_triple)] = tuple(gr_triple + ('ground_truth',))
    #     else:
    #         recontructed_graph.append(tuple(gr_triple + ('disappeared',)))
    ###

    ###
    recontructed_graph = []
    entity = list(tf.entity_id_to_label.values())
    n_entity = len(entity)
    for i in range(n_entity):
        for j in range(n_entity):
            if i != j:
                subject = entity[i]
                object = entity[j]

                pred = predict_target(
                    model = result.model,
                    head = subject,
                    tail = object,
                    triples_factory = result.training,
                )

                pred_df = pred.df
                score_softmax = scipy.special.softmax(pred_df['score'], axis=None)
                pred_df['score_softmax'] = score_softmax
                
                # predict new triple
                for _, row in pred_df[pred_df['score_softmax'] >= CONFIDENCE_THRESHOLD].iterrows():
                    if len(df.loc[(df['subject']==subject) & (df['predicate']==row['relation_label']) & (df['object']==object)]) >= 1:
                        recontructed_graph.append((subject, row['relation_label'], object, 'Ground Truth'))
                    else:
                        recontructed_graph.append((subject, row['relation_label'], object, 'Predict'))

                for _, row in df.iterrows():
                    found = False
                    for triple in recontructed_graph:
                        if triple[0] == row['subject'] and triple[1] == row['predicate'] and triple[2] == row['object']:
                            found = True
                        
                    if not found:
                        recontructed_graph.append((row['subject'], row['predicate'], row['object'], 'Disappeared'))
    ###

    return recontructed_graph

