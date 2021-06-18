import dill


def percentage_change(org, new):
    return (new - org) / org * 100


path = './'

# open result file of perturbations
with open(path, 'rb') as f:
    data = dill.load(f)

    scores = {}

    # loop through segmentation technqiues
    for key in data:

        scores[key] = {}

        # loop through perturbation methods
        for method in data[key]:
            if data[key][method] is None:
                continue

            d = data[key][method]

            org = d['original']
            per = d['percentile']
            rnd = d['random']
            score = percentage_change(org, per) / percentage_change(org, rnd)

            scores[key][method] = score

    print(scores)
