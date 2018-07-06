import numpy

def norm_fn1(x):
    return (x[1]-mean)/stdev

def norm_fn2(x):
    return (x[1]/mean**0.5)**0.25

def norm_fn3(x):
    return numpy.log(x[0]*x[1]*1.0/mean)

def norm_fn4(x):
    return numpy.log(x[0]*x[2]*100.0/mean)

def norm_fn5(x):
    return numpy.log(x[0]*max_cases/mean)

def takeRate(ele):
    if ele[2]==0:
        return -9999
    return ele[2]*ele[3]

dat1 = [[1,1],[5,5],[29,30],[99,100],[50,99],[20,99],[1,80]]

dat = [[69,103],[46,70],[28,48],[18,22],[1,1],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]

totCases=[x[1] for x in dat]
mean=sum(totCases)/len(totCases)
max_cases=max(totCases)*1.0 
stdev = numpy.std(totCases)

for m in dat:
    if m[1]==0:
        m.append(0)
    else:
        m.append((m[0]*100.0/m[1]))
    if m[2]==0:
        m.append(-9999)
    else:
        m.append(norm_fn5(m))

dat.sort(key=takeRate, reverse=True)
i=1
for m in dat:
    print 'Rank:', i, 'Total Cases:', m[1], 'Rate:', m[2], 'Measure:', m[3]
    i+=1
