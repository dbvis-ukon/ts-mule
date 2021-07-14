import dill

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Perturbation change towards random perturbation.')
    parser.add_argument('-path', type=str, default='', nargs='?', help='path to the perturbation file')

    args = parser.parse_args()
    path = args.path

    for f in listdir(path):
        if isfile(join(path, f)) and 'perturbation' in f:

            # open result file of perturbations
            with open(join(path, f), 'rb') as f:
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
                        score = [
                            (per - org) / (rnd - org),
                            (per - rnd) / (rnd - org),
                            per - org,
                            rnd - org
                        ]

                        scores[key][method] = score

                print('-' * 15)
                print(f)
                print(scores)
