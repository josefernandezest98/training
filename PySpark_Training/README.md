![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/apache_spark.webp](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/apache_spark.webp)

# PySpark Training Notebook

The idea of ​​this notebook is to show a first approximation of PySpark. It does not encompass all the functions, operators and modules, but rather offers a general idea of ​​how to use it and the syntax as an alternative to using the pandas and scikit-learn libraries, in cases of handling very large datasets. This notebook contains explanations of the methods alternated with exercises to practice.

## 0. Requirements

* functools.reduce
* pyspark.sql.SparkSession
* pyspark.sql.types.StructType, StructField, StringType, IntegerType
* pyspark.sql.window.Window
* pyspark.sql.functions.lit, concat, concat_ws, collect_set, collect_list, count, sum, avg, struct, col, to_date, lag, lead, udf, cume_dist, datediff, desc
* pandas
* numpy
* matplotlib.pyplot
* seaborn
* pyspark.ml.feature.VectorAssembler, Normalizer, StandardScaler, PCA
* pyspark.ml.functions.vector_to_array
* from pyspark.ml.clustering.KMeans
* pyspark.ml.evaluation.ClusteringEvaluator
* from statsmodels.distributions.empirical_distribution.ECDF
* scipy.stats.ks_2samp
* tensorflow

## 1. Introduction to Apache Spark, PySpark and distributed data 

Brief introduction to Apache Spark and PySpark, Resilient Distributed Datasets (RDD), PySpark DataFrames and PySpark SQL. In this part, we operate with a Python List and with a PySpark DataFrame. The main used methods are:

* map, reduce, createDataFrame, select, join, lit, withColumn, filter, drop, union, show, alias, toPandas, groupBy, agg, struct, col, concat, concat_ws

## 2. Data Analytics 

In this case, we download a public Kaggle dataset about the FIFA 17 players. We practice the main functions we have seen before. We continue a little with the idea of ​​dataset filtering:

* isNull, filter, groupBy, agg, sum, avg, pivot, to_date, Window, lag, lead, orderBy, udf, datediff, between, sort, desc

The second part of this section incorporates data plots:

* Histograms, pie charts, boxplots, correlation matrix, distribution and density plots

## 3. Machine Learning

This section is not a summary of the main machine learning techniques. It consists of two use case exercises applied to the FIFA dataset:

### Exercise 1: 

Neymar leaves FC Barcelona. Who is the most suitable football player to replace him?

* Filter the football players, who does not belong to FC Barcelona and are rated over 75.
* Transform categorical columns into numerical columns.
* PCA in order to reduce the number of columns. Before that, we should normalize and standardize the data.
* Clustering with Kmeans, number of clusters with Silhouette score.

### Exercise 2:

Luis Enrique wants to change teams from FC Barcelona to FC Bayern Munich, but wants to make sure both teams are similar. If not, give him any recommendation.

* Filter both teams, FC Barcelona and FC Bayern Munich.
* Two-sample Kolmogorov-Smirnov test. First, with scipy.stats and then, with PySpark submodules.
* Outliers detection in order to recommend which players should leave the team.

## 4. Deep Learning:
