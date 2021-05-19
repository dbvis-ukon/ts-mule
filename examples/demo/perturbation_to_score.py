import dill


def percentage_change(org, new):
    return (new - org) / org * 100

path = './'

with open(path, 'rb') as f:
    data = dill.load(f)
    
    scores = {}
    
    for key in data:
        
        scores[key] = {}
        
        for method in data[key]:
            if data[key][method] == None:
                continue
            
            d = data[key][method]
            
            org = d['original']
            per = d['percentile']
            rnd = d['random']
            score = percentage_change(org, per) / percentage_change(org, rnd)
            
            scores[key][method] = score
            
    print(scores)
