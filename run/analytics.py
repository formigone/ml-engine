from __future__ import division
import MySQLdb
import argparse
import os


def index_file(file_handle, subm_id, db, cur):
  firstLine = True
  batch = []
  for line in file_handle:
    if firstLine:
      firstLine = False
      continue
    filename, label = line.split(',')
    minibatch = '({}, {}, {})'.format(
      subm_id,
      '(select id from file where name = "{}")'.format(filename.strip()),
      '(select id from label where name = "{}")'.format(label.strip()),
    )

    if len(batch) == 100:
      sql = ','.join(batch)
      cur.execute(sql)
      db.commit()
      batch = []

    if len(batch) == 0:
      sql = 'insert into submissionParts (submissionId, fileId, labelId) values {}'.format(minibatch)
      batch.append(sql)

    if 0 < len(batch) < 100:
      batch.append(minibatch)


def get_db():
  return MySQLdb.connect(host='127.0.0.1', user='root', passwd='', db='ml')


def submit(filename, submission_id=43):
  db = get_db()
  cur = db.cursor()

  cur.execute('select count(*) as total from (select count(*) from submissionParts where submissionId = {} group by fileId) as dm'.format(submission_id))
  total = int(cur.fetchone()[0])
  print('Total records: {}'.format(total))

  with open(filename) as fh:
    firstLine = True
    matches = 0
    batch = []
    batch_size = 500
    for line in fh:
      if firstLine:
        firstLine = False
        continue
      filename, label = line.strip().split(',')
      minibatch = '(submissionId = {} and fileId = {} and labelId = {})'.format(
        submission_id,
        '(select id from file where name = "{}")'.format(filename),
        '(select id from label where name = "{}")'.format(label),
      )

      if len(batch) == batch_size:
        sql = ' or '.join(batch) + ' group by fileId'
        cur.execute('select count(*) as total from ({}) as dm'.format(sql))
        batch_matches = int(cur.fetchone()[0])
        matches = matches + batch_matches
        print('{}% => {}%'.format(
          float('{0:.4f}'.format(batch_matches / batch_size * 100)),
          float('{0:.4f}'.format(matches / total * 100)),
        ))
        batch = []

      if len(batch) == 0:
        sql = 'select count(*) as total from submissionParts where {}'.format(minibatch)
        batch.append(sql)

      if 0 < len(batch) < batch_size:
        batch.append(minibatch)
    return total, matches


def index():
  raise Exception('Option disabled')
  db = get_db()
  cur = db.cursor()

  dir = '../tensorflow-speech-recognition-challenge/submissions'
  files = os.listdir(dir)
  for file in files:
    _, _, score, model = file.split('-')
    score = float(score)
    model = model.strip()
    sql = 'insert into model (name) values ("{}")'.format(model)
    cur.execute(sql)
    db.commit()
    model_id = cur.lastrowid

    sql = 'insert into submission (modelId, score) values ({}, {})'.format(model_id, score)
    cur.execute(sql)
    db.commit()
    subm_id = cur.lastrowid

    with open('{}/{}'.format(dir, file)) as subm:
      print(file)
      index_file(subm, subm_id=subm_id, db=db, cur=cur)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--index', type=str, help='Index a submission file')
  parser.add_argument('--submit', type=str, help='Path to CSV to be submitted')

  args = parser.parse_args()

  if args.index:
    index()
  elif args.submit:
    total, matches = submit(args.submit)
    print('{}/{} total'.format(matches, total))
