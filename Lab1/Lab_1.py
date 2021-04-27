import gzip
import json
import ast
import sys

from pyspark import SparkContext, SparkConf

def pairRDD_map_5core(line):
	line_in_dict = ast.literal_eval(line)	
	return (line_in_dict["asin"], {line_in_dict["reviewerID"]})

def pairRDD_map_metadata(line):
	line_in_dict = ast.literal_eval(line)
	if "price" in line_in_dict:
		return (line_in_dict["asin"], [line_in_dict["price"]])
	else:
		return (line_in_dict["asin"], ["N/A"])

	
if __name__ == "__main__":	
	conf = SparkConf().setAppName("Lab_1").setMaster("local[*]")
	sc = SparkContext(conf = conf)

	review_rdd = sc.textFile(sys.argv[1])
	product_unique_review_rdd = review_rdd.map(lambda l : pairRDD_map_5core(l)).reduceByKey(lambda v1, v2: v1.union(v2)).mapValues(lambda v: len(v))

	metadata_rdd = sc.textFile(sys.argv[2])
	product_price_list_rdd =metadata_rdd.map(lambda l : pairRDD_map_metadata(l)).reduceByKey(lambda v1, v2: v1 + v2)

 	product_join_rdd = product_unique_review_rdd.join(product_price_list_rdd)
    	sorted_product_join_rdd = product_join_rdd.sortBy(lambda element: element[1][0], ascending=False)
	sorted_product_join_rdd_10 = sorted_product_join_rdd.take(10)

	rdd_10 = sc.parallelize(sorted_product_join_rdd_10)
	rdd_10.map(lambda l: "<{}> <{}> <{}>".format(l[0],l[1][0],l[1][1])).coalesce(1).saveAsTextFile(sys.argv[3])

	sc.stop()

	


