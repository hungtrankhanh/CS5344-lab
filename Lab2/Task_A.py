import math
import os
import re
import sys

from pyspark import SparkContext, SparkConf


def tf_idf_calculation(n, data):
    k = data[0]
    v = data[1]
    tf_idf = tf_idf_formula(v[0], v[1], n)
    return k[1], (k[0], tf_idf)


def tf_idf_formula(tf, df, n):
    return (1 + math.log10(tf)) * math.log10(n / df)


def flat_n_document_v2(data):
    k = data[0]
    head, tail = os.path.split(k)
    # v = re.split(r'[^\w]+', data[1])
    v = re.split(r'\s|(?<!\d)[,.]|[,.](?!\d)', data[1])

    vl = []
    for w in v:
        lower_w = w.lower()
        vl.append(((lower_w, tail), 1))

    return vl


def flat_n_document_v3(data):
    w = data[0][0]
    doc = data[0][1]
    numofwords = data[1]

    w2 = re.sub('[^A-Za-z0-9]+', '', w)

    return ((w2, doc), numofwords)


def valid_word_in_query(data, query_len):
    k = data[0]
    if k in query_rdd_collection:
        return k, 1/query_len
    else:
        return k, 0


if __name__ == "__main__":
    conf = SparkConf().setAppName("Lab_2_TaskA").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    accum = sc.accumulator(0)
    query_rdd = sc.textFile(sys.argv[1])
    query_rdd_collection = query_rdd.flatMap(lambda l: re.split(r'\s|(?<!\d)[,.]|[,.](?!\d)', l)) \
        .filter(lambda l1: l1 is not "") \
        .collect()

    #print("n_document_wd_n:", query_rdd_collection)

    stopword_rdd = sc.textFile(sys.argv[2])
    stopword_rdd_collection = stopword_rdd.collect()

    n_document = sc.wholeTextFiles(sys.argv[3])
    num_of_doc = n_document.count()
    n_document_wd_n = n_document.flatMap(flat_n_document_v2) \
        .filter(lambda w1: w1[0][0] not in stopword_rdd_collection) \
        .map(flat_n_document_v3) \
        .filter(lambda w2: w2[0][0] is not "" and w2[0][0] not in stopword_rdd_collection) \
        .reduceByKey(lambda v1, v2: v1 + v2)  # ((w,d), n)

    n_document_w_d = n_document_wd_n.map(lambda data2: data2[0])  # get (w,d)
    n_document_w_m = n_document_w_d.groupByKey().map(lambda data3: (data3[0], len(data3[1])))  # get (w,m)

    n_document_wd_n = n_document_w_d.join(n_document_w_m).map(lambda t: ((t[0], t[1][0]), t[1][1]))  # ((w,d), m)
    n_document_wd_nm = n_document_wd_n.join(n_document_wd_n)  # ((w,d), (n,m))
    n_document_d_wtf = n_document_wd_nm.map(lambda t: tf_idf_calculation(num_of_doc, t))  # (d,(w,tf))
    n_document_d_l = n_document_d_wtf.map(lambda t1: (t1[0], t1[1][1] * t1[1][1])) \
        .reduceByKey(lambda n1, n2: (n1 + n2)) \
        .map(lambda t2: (t2[0], math.sqrt(t2[1])))  # (d,l)

    n_document_d_wtf_l = n_document_d_wtf.join(n_document_d_l)  # (d, ((w,tf), l))
    n_document_d_wnorm = n_document_d_wtf_l.map(lambda t: (t[0], (t[1][0][0], t[1][0][1] / t[1][1])))  # (d,(w,norm))
    n_document_w_dnorm = n_document_d_wnorm.map(lambda t : (t[1][0], (t[0], t[1][1]))) # (w,(d,d_norm))
    '''
    n_document_d_lnorm = n_document_d_wnorm.map(lambda t: (t[0], t[1][1] * t[1][1])) \
        .reduceByKey(lambda n1, n2: (n1 + n2)) \
        .map(lambda data: (data[0], math.sqrt(data[1])))  # (d, lnorm)
    
    n_document_d_wnorm_lnorm = n_document_d_wnorm.join(n_document_d_lnorm)  # (d,((w,norm),lnorm))
    n_document_w_dnormlnorm = n_document_d_wnorm_lnorm.map(
        lambda t: (t[1][0][0], (t[0], t[1][0][1], t[1][1])))  # (w, (d, norm, lnorm))
    '''
    n_query_w_n = n_document_w_m.filter(
        lambda l: l[0] in query_rdd_collection)  # find words in query, if it is found, then it is asigned to 1
    n_query_len = math.sqrt(n_query_w_n.count())

    n_query_w_nl = n_document_w_m.map(
        lambda l: valid_word_in_query(l, n_query_len))  # calculate magnitude of query (w, q_norm))

    n_document_query = n_document_w_dnorm.join(n_query_w_nl)  # (w,((d, d_norm), q_norm))
    n_document_query2 = n_document_query.map(
        lambda t: (t[1][0][0], t[1][0][1] * t[1][1]))  # (d, product)
    n_document_query3 = n_document_query2.reduceByKey(
        lambda v1, v2: v1 + v2)  # (d,sum of product)

    n_document_query_relevant_sorted = n_document_query3.sortBy(lambda e: e[1], ascending=False)
    top_10_document_relevant = n_document_query_relevant_sorted.take(10)

    rdd_10 = sc.parallelize(top_10_document_relevant)
    rdd_10.map(lambda l: "<{}> <{}>".format(l[0], l[1])).coalesce(1).saveAsTextFile(sys.argv[4])
    sc.stop()
