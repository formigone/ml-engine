import sys

file = sys.argv[1]

with open(file) as fh:
  for line in fh:
    line = line.strip()
    parts = line.split(',')[:2]
    print(','.join(parts))
