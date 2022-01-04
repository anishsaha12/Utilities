from multiprocessing import Pool

def apply_function_on_chunk(list_items):
    records = []
    for i,list_item in enumerate(list_items):
        ## Do processing on list_item
        records.append(list_item) 
    return records

def f_mp(a_list, size):
    chunks = [a_list[i::size] for i in range(size)]
    pool = Pool(processes=size)
    result = pool.map(apply_function_on_chunk, chunks)
    return result

items = [0,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]

records = f_mp(items, 30)

f_records = []
for r in records:
    f_records = f_records+r
    
len(f_records)  # result
