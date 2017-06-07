import sys, re, pdb, random
from nltk.stem import SnowballStemmer

stem = SnowballStemmer('english').stem
grep_tup = re.compile(r'(.+?), \((\S+), (\S+)\), (-*1)').findall

# OK!
def read_input(inp):
    with open(inp, 'r+') as fin:
        text = fin.read()
    return text

# OK!
def process_feat(f):         # called by import_goldstandard only
    return f.lower()
    #return stem(f.lower())

# OK!
def import_goldstandard(inptrue):                   # the goldstandard imported is not following the order as in goldstandard.txt
    global votes, goldstandard, goldstandard_tnum, goldstandard_str, multifeatgps
    tmp = read_input(inptrue).split('\n\n')
    tmp = [record.strip().split('\n') for record in tmp] # tmp = [['A1WSO7X5LNEMRH B0007V8DS8 5.0 0.5668658561714404','phone, (good, positive), 1', ....], ....]
    votes = [records[0] for records in tmp]             # votes only store reviewId to ntime, e.g. votes = ['A1WSO7X5LNEMRH B0007V8DS8 5.0 0.5668658561714404', ...]
    goldstandard = {}
    multifeatgps = {}
    goldstandard_tnum = {}
    goldstandard_str = {}
    insuccess = []
    for records in tmp:
        vote = records[0]
        if vote in goldstandard: print(vote)
        goldstandard[vote] = {}
        goldstandard_tnum[vote] = 0
        multifeatgps[vote] = []
        goldstandard_str[vote] = [record for record in records[1:] if record]           # goldstandard_str[vote] = ['aspect1', 'aspect2']
        for record in records[1:]:
            try: tuplet = grep_tup(record)[0]                   # if dont match, put in insuccess
            except: insuccess.append((vote,record))
            if '/' not in tuplet[0]: 
                pfeat = process_feat(tuplet[0])
                goldstandard[vote].setdefault(pfeat, {}).update({tuplet[1].lower():(tuplet[2], tuplet[3])})
                goldstandard_tnum[vote] += 1
            else:
                pfeats = list(map(process_feat, tuplet[0].split('/')))
                multifeatgps[vote].append((pfeats, tuplet[1]))
                for pfeat in pfeats:
                    goldstandard[vote].setdefault(pfeat, {}).update({tuplet[1].lower():(tuplet[2], tuplet[3])})
                    goldstandard_tnum[vote] += 1
    print('Gold standard imported; %d records could not be recognized' % (len(insuccess)))
    for vote, record in insuccess:
        print('\t%s: %s' % (vote, record))

# OK!
def import_result(inpresult):               # note that result has size = 100 only! not 2000, it will only import the result review from output_sentiment.txt which is in goldstandard.txt
    global result, trainsize
    tmp = read_input(inpresult).split('\n\n')
    tmp = [record.strip().split('\n') for record in tmp]
    trainsize = len(tmp)
    result = {}
    for vote in tmp:
        key = ' '.join(vote[0].split()[:-1])
        if key not in goldstandard: continue
        result[key] = []
        rappend = result[key].append
        for record in vote[1:]:
            tuplet = grep_tup(record)[0]
            rappend(tuplet)

# 
def calculate_stat(vote, rtuplets):             # called by update_stat_by_votes only     # remeber rtuplets is not from goldstandard
    global result_str
    result_str[vote] = []
    sappend = result_str[vote].append
    ttuplets = goldstandard[vote]               # ttuplets = {'phone':{'good':('positive',1)}}   or {'phone':{'good':(...), 'great':(...)}
                                                # goldstandard_tnum = {'A1234 ....': 1},  where 1 is # of hand written aspect/column, despite same aspect
    gopnum = goldstandard_tnum[vote]            # for that vote in goldstandard, # of hand written aspect/column
    ropnum = len(rtuplets)                      # rtuplets is the parameter!
    tfeat = set(ttuplets.keys())                # for 1 tfeat = {aspect1, aspect2,...}  and DISTNCT!
    rfeat = set([tuplet[0] for tuplet in rtuplets])     # similar to tfeat, but rfeat are the aspects from result
    cfeatnum = len(rfeat & tfeat)               # find intersection
    opcfeatnum = sum([len(ops) for feat, ops in ttuplets.items() if feat in rfeat])
    copfeatnum = 0
    copnum = 0
    csentnum = 0
    cnegnum = 0
    for feat, op, sent, neg in rtuplets:
        if feat not in ttuplets: 
            sappend('(%s, (%s, %s), %s)'%(feat, op, sent, neg))
            continue
        copfeatnum += 1
        if op not in ttuplets[feat]: 
            sappend('(*%s, (%s, %s), %s)'%(feat, op, sent, neg))
            continue
        copnum += 1
        if sent == ttuplets[feat][op][0]: 
            sent = '*'+sent
            csentnum += 1
        if neg == ttuplets[feat][op][1]: 
            neg = '*'+neg
            cnegnum += 1
        sappend('(*%s, (*%s, %s), %s)'%(feat, op, sent, neg))

    for featgp, op in multifeatgps[vote]:
        excess = [feat for feat in featgp if any(rfeat == feat and rop == op for rfeat, rop, rsent, rneg in rtuplets)]
        if excess: gopnum -= len(featgp) - len(excess)
    
    return ((cfeatnum, len(rfeat)) if len(rfeat) != 0 else (1 if len(tfeat) == 0 else 0), 
            (cfeatnum, len(tfeat)) if len(tfeat) != 0 else 1,
            (copnum, ropnum) if ropnum != 0 else (1 if gopnum == 0 else 0), 
            (copnum, gopnum) if gopnum != 0 else 1,
            (copnum, copfeatnum) if copfeatnum != 0 else None,
            (copnum, opcfeatnum) if opcfeatnum != 0 else None, 
            (csentnum, ropnum) if ropnum != 0 else (1 if gopnum == 0 else 0),
            (csentnum, gopnum) if gopnum != 0 else 1, 
            (csentnum, copnum) if copnum != 0 else None,
            (cnegnum, ropnum) if ropnum != 0 else (1 if gopnum == 0 else 0),
            (cnegnum, gopnum) if gopnum != 0 else 1,
            (cnegnum, copnum) if copnum != 0 else None)

# OK!
def update_stat_by_votes():
    global stat_by_votes, result_str
    result_str = {}
    stat_by_votes = {vote:calculate_stat(vote, rtuplets) for vote, rtuplets in result.items()}          # result.items() = [('A12344 ...', [(aspect, (opinion, positive), 1), (...)]), ('A1235677'....)]
    # vote = 'A123455...'
    # rtuples = [(aspect1, (adj1, positive), 1),(...),...]  it is not from goldstandard!!!!!!! not hand written
    # stat_by_votes = {'A123 ....':((1,4),(1,4),None,...)}    => means A123... this vote has fp = 1/4, fr = 1/4, opp = None = '-'

# 
def stat_by_scores():     # called by final print_output only
    return [[scores[i] for vote, scores in stat_by_votes.items()] for i in range(12)]

#
def avg_with_None(l):     # called by final print_output only
    tmp = [i if type(i)!=tuple else i[0]/i[1] for i in l if i != None]
    """
    if len(tmp) == 0:
        x = 0
    else:
        x = sum(tmp)/len(tmp)
    return (x, len(tmp))
    """
    return (sum(tmp)/len(tmp), len(tmp))

# 
def val_to_str(val):      # called by final print_output only
    if val == None: return '-'
    if type(val) == int: return str(val)
    if type(val) == float: return '%.5f'%(val)
    return '%d/%d'%(val[0], val[1])

def print_output(outp):
    header = ['fp', 'fr','opp', 'opr', 'opcp', 'opcr', 'sp', 'sr', 'sc', 'np', 'nr', 'nc', 'vote']
    with open(outp, 'w+') as fout:
        fwrite = fout.write
        remark = input('Remark on this set of input (end by return): ')
        fwrite('training set size: %d\nevaluate set size: %d\nremark: %s\n\n' % (trainsize, len(result), remark))
        fwrite('Average scores:\n')
        fwrite('\t' + ''.join(['{:10}'.format(v) for v in header[:-1]]) + '\n')
        fwrite('\t' + ''.join(['{:10}'.format(val_to_str(v[0])) for v in map(avg_with_None, stat_by_scores())]) + '\n')
        fwrite('\nScores for each review:\n')
        fwrite('\t' + ''.join(['{:10}'.format(v) for v in header]) + '\n')
        fwrite('\n'.join(['{:<4}'.format(i+1) + ''.join([('{:10}').format(val_to_str(v)) for v in stat_by_votes[vote]]) + '%s'%(vote) for i, vote in enumerate(votes)]))
        if detailed:
            fwrite('\n\n')
            fwrite('\n\n'.join(['%s\ngold standard: %s\nextracted: %s'%(vote, ', '.join(['(%s)'%(s) for s in goldstandard_str[vote]]), ', '.join(result_str[vote])) for vote in votes]))

# OK!
def create_goldstandard(inpresult, evalsize, outp):
    votes = [' '.join(record.strip().split('\n')[0].split()[:-1]) for record in read_input(inpresult).split('\n\n')]
    evalset = sorted(random.sample(range(len(votes)), evalsize))
    with open(outp, 'w+') as fout:
        fout.write('\n\n'.join([votes[i] for i in evalset]))

# 
def main(inptrue, inpresult, outp, detailed = False):
    import_goldstandard(inptrue)
    print('Imported gold standard, evaluation set size: %d' % (len(goldstandard)))
    import_result(inpresult)
    print('Imported result, number of review records matching evaluation set: %d' % (len(result)))
    if len(result) != len(goldstandard):
        print('Error: Missing votes for evaluation; votes in gold standard but not in result:\n%s' % (', '.join([key for key in goldstandard if key not in result])))
        exit()
    update_stat_by_votes()
    print_output(outp)

# OK!
if __name__ == '__main__':
    global detailed
    detailed = False
    if len(sys.argv) > 5: detailed = True if sys.argv[5] == 'detailed' else False       # > 5 means running eval   global variable detailed = true only if running eval and detailed argv = detailed
    if sys.argv[1] == 'eval': main(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'init': create_goldstandard(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else: print('Usage: python3.4 evaluate_sentiment.py eval <goldstandard> <result> <output> (opt: detailed)\n       python3.4 evaluate_sentiment.py init <result> <evalset size> <output>')
    # else user not inputting eval or init