#............................................................................
#...............SSSSSS.......................................................
#.TTTTTTTTTTT..SSSSSSSS...MMMMM...MMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
#.TTTTTTTTTTT.SSSSSSSSSS..MMMMM...MMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
#.TTTTTTTTTTT.SSSSSSSSSS..MMMMMM..MMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
#.....TTTT...SSSS...SSSSS.MMMMMM.MMMMMM..UUUU...UUUU..LLLL.......EEEE........
#.....TTTT...SSSSSS.......MMMMMM.MMMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
#.....TTTT....SSSSSSSSS...MMMMMM.MMMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
#.....TTTT....SSSSSSSSSS..MMMMMM.MMMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
#.....TTTT......SSSSSSSSS.MMMMMMMMMMMMM..UUUU...UUUU..LLLL.......EEEEEEEEEE..
#.....TTTT...SSSS..SSSSSS.MMM.MMMMMMMMM..UUUU...UUUU..LLLL.......EEEE........
#.....TTTT...SSSS....SSSS.MMM.MMMMM.MMM..UUUU...UUUU..LLLL.......EEEE........
#.....TTTT...SSSSSSSSSSSS.MMM.MMMMM.MMM..UUUUUUUUUUU..LLLLLLLLLL.EEEEEEEEEE..
#.....TTTT....SSSSSSSSSS..MMM.MMMMM.MMM..UUUUUUUUUUU..LLLLLLLLLL.EEEEEEEEEE..
#.....TTTT.....SSSSSSSSS..MMM.MMMMM.MMM...UUUUUUUUU...LLLLLLLLLL.EEEEEEEEEE..
#...............SSSSSS......................UUUUU............................
#............................................................................
"""
tsmule 
~~~~~~~
tsmule is an explainable AI library, written in Python, for time series explainations.
TODO: change this following usages

Basic GET usage:
   >>> import tsmule
   >>> r = requests.get('https://www.python.org')
   >>> r.status_code
   200
   >>> b'Python is a programming language' in r.content
   True
... or POST:
   >>> payload = dict(key1='value1', key2='value2')
   >>> r = requests.post('https://httpbin.org/post', data=payload)
   >>> print(r.text)
   {
     ...
     "form": {
       "key1": "value1",
       "key2": "value2"
     },
     ...
   }
   
The other HTTP methods are supported - see `requests.api`. Full documentation
is at <https://requests.readthedocs.io>.
:copyright: (c) 2017 by Kenneth Reitz.
:license: Apache 2.0, see LICENSE for more details.
"""

# from .segment import MatrixProfileSegmentation, SAXSegmentation
# from .perturb import Perturbation

# __all__ = ['Perturbation', 
#            'MatrixProfileSegmentation',
#            'SAXSegmentation'
#            ]
