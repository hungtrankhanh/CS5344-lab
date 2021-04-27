import math
import os
import sys
import re

from pyspark import SparkContext, SparkConf


class K_Mean:
    def __init__(self, kmean, words, datapoints, dist_type):
        self._kmean = kmean
        self._wordlist = words
        self._datapoints = datapoints
        self._centeroids = [0 for x in range(kmean)]
        self._cluster_group = {}
        self._dist_type = dist_type

    # [(d,[(w,norm), (w,norm)], ()
    def initialize_centeroids(self):
        keys = list(self._datapoints.keys())
        jump_idx = math.floor(len(self._datapoints) / self._kmean)
        for i in range(self._kmean):
            key_index = i + int(jump_idx)
            key = keys[key_index]
            print("initialize_centeroids : k=", self._kmean, " doc size=", len(self._datapoints), " file=", key)
            self._centeroids[i] = self._datapoints[key]

    def DistanceCalculation(self, x, y):
        if self._dist_type == "Cosine_Similarity":
            return self.CosineDistance(x, y)
        if self._dist_type == "Euclidean":
            return self.EuclideanDistance(x, y)

    def CosineDistance(self, x, y):
        square_sum_x = 0
        square_sum_y = 0
        product_x_y = 0
        for i in self._wordlist:
            x_v = y_v = 0
            if i in x:
                x_v = x[i]
            if i in y:
                y_v = y[i]
            product_x_y += x_v * y_v
            square_sum_x += x_v * x_v
            square_sum_y += y_v * y_v

        consine_similarity = product_x_y / (math.sqrt(square_sum_x) * math.sqrt(square_sum_y))
        return 1 - consine_similarity

    def EuclideanDistance(self, x, y):
        distance = 0
        for i in self._wordlist:
            x_v = y_v = 0
            if i in x:
                x_v = x[i]
            if i in y:
                y_v = y[i]

            distance += math.pow(x_v - y_v, 2)
        return math.sqrt(distance)

    def find_centeroid_For(self, x):
        minDistance = -1
        minIndex = -1
        for i in range(self._kmean):
            if minDistance == -1:
                minDistance = self.DistanceCalculation(x, self._centeroids[i])
                minIndex = i
            else:
                temp = self.DistanceCalculation(x, self._centeroids[i])
                if minDistance > temp:
                    minDistance = temp
                    minIndex = i
        return minIndex

    def kmean_running(self, path):
        key_vector_list = list(self._datapoints.keys())
        start_running = True
        running_index = 0
        while start_running:
            check_kmean_stop = True
            for key in key_vector_list:
                value_dic = self._datapoints[key]
                centroid_idx = self.find_centeroid_For(value_dic)
                if key in self._cluster_group:
                    key_idx = self._cluster_group[key]
                    if centroid_idx != key_idx:
                        check_kmean_stop = False
                        self._cluster_group[key] = centroid_idx
                else:
                    check_kmean_stop = False
                    self._cluster_group[key] = centroid_idx

            self.save_cluster_to_file(running_index, path)

            running_index += 1

            if check_kmean_stop:
                start_running = False
            self.update_centeroids()

    def update_centeroids(self):
        for i in range(self._kmean):
            item_list = []
            cluster_keys = list(self._cluster_group.keys())
            for ck in cluster_keys:
                if i == self._cluster_group[ck]:
                    item_list.append(ck)
            dic = {}
            item_len = len(item_list)
            for el in item_list:  # filename
                item_dic = self._datapoints[el]
                for w in self._wordlist:
                    dic_v = 0
                    item_dic_v = 0
                    if w in dic:
                        dic_v = dic[w]
                    if w in item_dic:
                        item_dic_v = item_dic[w]

                    dic[w] = dic_v + item_dic_v
            for w in self._wordlist:
                dic[w] = dic[w] / item_len
            self._centeroids[i] = dic

    def save_cluster_to_file(self, loop_index, path):
        output2 = os.path.join(path, "{}_k{}_iter{}.txt".format(self._dist_type, self._kmean, loop_index))
        cluster_keys = list(self._cluster_group.keys())
        with open(output2, 'w') as f:
            for k in range(self._kmean):
                group = []
                for key in cluster_keys:
                    if k == self._cluster_group[key]:
                        group.append(key)
                f.write("group_{} : {}\n".format(k, group))


def tf_idf_calculation(n, data):
    # ((w,d), (n,m))
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


def parse_collection_to_dataset(collection):
    vectors = {}
    for item_n in collection:
        dic = {}
        # print("item: ", item[0])
        for value in item_n[1]:
            # print("value 1: ", value[0], "value 2:", value[1])
            dic[value[0]] = value[1]
        vectors[item_n[0]] = dic
    return vectors


if __name__ == "__main__":
    conf = SparkConf().setAppName("Lab_2_TaskB").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    stopword_rdd = sc.textFile(sys.argv[1])
    stopword_rdd_collection = stopword_rdd.collect()

    n_document = sc.wholeTextFiles(sys.argv[2])
    num_of_doc = n_document.count()
    n_document_wd_n = n_document.flatMap(flat_n_document_v2) \
        .filter(lambda w1: w1[0][0] not in stopword_rdd_collection) \
        .map(flat_n_document_v3) \
        .filter(lambda w2: w2[0][0] is not "" and w2[0][0] not in stopword_rdd_collection) \
        .reduceByKey(lambda v1, v2: v1 + v2)  # ((w,d), n)

    n_document_w_d = n_document_wd_n.map(lambda data2: data2[0])  # get (w,d)
    n_document_w_m = n_document_w_d.groupByKey().map(lambda data3: (data3[0], len(data3[1])))  # get (w,m)
    n_document_words = n_document_w_m.map(lambda t: t[0])

    n_document_wd_n = n_document_w_d.join(n_document_w_m).map(lambda t: ((t[0], t[1][0]), t[1][1]))  # ((w,d), m)
    n_document_wd_nm = n_document_wd_n.join(n_document_wd_n)  # ((w,d), (n,m))
    n_document_d_wtf = n_document_wd_nm.map(lambda t: tf_idf_calculation(num_of_doc, t))  # (d,(w,tf))
    n_document_d_l = n_document_d_wtf.map(lambda t1: (t1[0], t1[1][1] * t1[1][1])) \
        .reduceByKey(lambda n1, n2: (n1 + n2)) \
        .map(lambda t2: (t2[0], math.sqrt(t2[1])))  # (d,l)

    n_document_d_wtf_l = n_document_d_wtf.join(n_document_d_l)  # (d, ((w,tf), l))
    n_document_d_wnorm = n_document_d_wtf_l.map(lambda t: (t[0], (t[1][0][0], t[1][0][1] / t[1][1])))  # (d,(w,norm))

    n_document_d_wnorm2 = n_document_d_wnorm.groupByKey()
    n_document_collection = n_document_d_wnorm2.collect()
    word_list = n_document_words.collect()
    data_points = parse_collection_to_dataset(n_document_collection)

    if not os.path.exists(sys.argv[3]):
        os.makedirs(sys.argv[3])

    for kn in range(2, 9):
        k_mean1 = K_Mean(kn, word_list, data_points, "Cosine_Similarity")
        k_mean1.initialize_centeroids()
        k_mean1.kmean_running(sys.argv[3])

        k_mean2 = K_Mean(kn, word_list, data_points, "Euclidean")
        k_mean2.initialize_centeroids()
        k_mean2.kmean_running(sys.argv[3])

    sc.stop()
