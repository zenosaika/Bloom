from django.shortcuts import render

import pandas as pd
import numpy as np
import json

def link_prediction(request):
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
            'target': row['object']
        })
    
    return json.dumps(graph)