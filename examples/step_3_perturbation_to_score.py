import argparse
from os import listdir
from os.path import isfile, join

import dill


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Perturbation change towards random perturbation.')
    parser.add_argument('-path', type=str, default='', nargs='?', help='path to the perturbation file')

    args = parser.parse_args()
    path = args.path

    for filename in listdir(path):
        full_path = join(path, filename)
        if isfile(full_path) and 'perturbation' in filename:

            # open result file of perturbations
            with open(full_path, 'rb') as fd:
                data = dill.load(fd)

                scores = {}

                # loop through segmentation techniques
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
                        denom = rnd - org
                        if denom == 0:
                            score = [float('nan'), float('nan'), per - org, rnd - org]
                        else:
                            score = [
                                (per - org) / denom,
                                (per - rnd) / denom,
                                per - org,
                                rnd - org,
                            ]

                        scores[key][method] = score

                print('-' * 15)
                print(filename)
                print(scores)
