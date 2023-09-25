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

* Histograms

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/hist.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/hist.png)

* Pie chart

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/pie.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/pie.png)

* Boxplots

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/boxplot.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/boxplot.png)

* Correlation matrix

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/corr.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/corr.png)

* Distribution plot

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/distribution.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/distribution.png)

* Density plot

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/density.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/density.png)

* Pair plots

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/pairs.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/pairs.png)

## 3. Machine Learning

This section is not a summary of the main machine learning techniques. It consists of two use case exercises applied to the FIFA dataset:

### Exercise 1: 

Neymar leaves FC Barcelona. Who is the most suitable football player to replace him?

* Filter the football players, who does not belong to FC Barcelona and are rated over 75.
* Transform categorical columns into numerical columns.
* PCA in order to reduce the number of columns. Before that, we should normalize and standardize the data.

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/pca.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/pca.png)

* Clustering with Kmeans, number of clusters with Silhouette score.

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/sil.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/sil.png)

We have decided Kmeans randomly, but we could have used any other clustering technique.

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/clustering.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/clustering.png)

### Exercise 2:

Luis Enrique wants to change teams from FC Barcelona to FC Bayern Munich, but wants to make sure both teams are similar. If not, give him any recommendation.

* Filter both teams, FC Barcelona and FC Bayern Munich.
* Two-sample Kolmogorov-Smirnov test. First, with scipy.stats and then, with PySpark submodules.

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/distributions.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/distributions.png)
![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/cums.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/cums.png)

* Outliers detection in order to recommend which players should leave the team.

## 4. Deep Learning:

This last module tries to integrate a the gradient descent algorithm with tensorFlow and Pandas. Gradient descent algorithm does not work for all functions. There are two specific requirements. A function has to be:

*  differentiable: a function is differentiable if it has a derivative for each point in its domain.
*  convex: for two points $x_1, x_2$ the function is convex if for $\lambda \in \left[0,1\right]$ the points verify

$$f\left(\lambda x_1 + \left(1 - \lambda\right)x_2\right) \leq \lambda f\left(x_1\right) + \left(1 - \lambda\right)f\left(x_2\right).$$

A gradient for an n-dimensional function f(x) at a given point p is defined as follows:

$$\nabla f\left(p\right) = \left[\frac{\partial f}{\partial x_1}\left(p\right), \dots, \frac{\partial f}{\partial x_n}\left(p\right)\right]^\top. $$

In particular, we want to optimize de parameters of Gompertz function:

$$f\left(t\right) = b_0 e^{-b_1 e^{-b_2 t}},$$

where:

*  $b_0$ is an asymptote, since $\lim_{t\to\infty} b_0 e^{-b_1 e^{-b_2 t}} = b_0 e^0 = b_0.$
*  $b_1$ sets the displacement along the x-axis.
*  $b_2$ sets the growth rate (*y* scaling).
*  $e$ is Euler´s Number $e \approx 2.71828\dots$

in GADA derivation shape:

$$f\left(t_0,y_0,t,b_1,b_2\right) = b_1 e^{\left[ \left(\frac{\log\left(\frac{y_0}{b_1}\right)}{e^{-b_2 t_0}}\right)e^{-b_2 t} \right]}.$$

![https://raw.githubusercontent.com/josefernandezest98/training/blob/main/PySpark_Training/gompertz.png](https://raw.githubusercontent.com/josefernandezest98/training/main/PySpark_Training/gompertz.png)

## 5. References

[1] [Udacity: Data Engineer Nanodegree](https://www.udacity.com/course/data-engineer-nanodegree--nd027?gclid=CjwKCAjwrranBhAEEiwAzbhNtWNuoqrgp10cpxGR83B9ZhKLR_j0EBx6gSDXISPxaZ3PKWulrSQHchoC2VcQAvD_BwE&utm_campaign=19167921312_c_individuals&utm_keyword=udacity%20data%20engineering_e&utm_medium=ads_r&utm_source=gsem_brand&utm_term=143524475719)

[2] [Kaggle: Advanced PySpark for Exploratory Data Analysis](https://www.kaggle.com/code/tientd95/advanced-pyspark-for-exploratory-data-analysis/notebook#5.-Explolatory-Data-analysis-)

[3] [Data Analysis: What is Data Analytics?](https://careerfoundry.com/en/blog/data-analytics/what-is-data-analytics/?utm_campaign=156497489670&utm_term=what%20is%20data%20analytics&utm_source=google&utm_medium=cpc&utm_content=670526673808&hsa_acc=9974869296&hsa_cam=6806097117&hsa_grp=156497489670&hsa_ad=670526673808&hsa_src=g&hsa_tgt=kwd-29089410432&hsa_kw=what%20is%20data%20analytics&hsa_mt=p&hsa_net=adwords&hsa_ver=3&gclid=CjwKCAjwrranBhAEEiwAzbhNtRWWrYfq6uDNetxWoMAI_yNrVJXhc9_UqUiixvCEtnSug0ujhcVd5BoC4jYQAvD_BwE)

[4] [K Means: Algorithm, Applications, Evaluation Methods, and Drawbacks](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a)

[5] [How to Use Silhouette Score in K-Means](https://saturncloud.io/blog/how-to-use-silhouette-score-in-kmeans-clustering-from-scikitlearn-library/#:~:text=The%20silhouette%20score%20is%20a%20useful%20metric%20for%20evaluating%20the,is%20from%20the%20neighboring%20clusters.)

[6] [Scikit-Learn Documentation: Clustering Techniques Comparison in 2D](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)

[7] [Wikipedia: Gompertz Curve Model](https://en.wikipedia.org/wiki/Gompertz_function)

[8] [Towards Data Science: Gradient Descent Algortihm](https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21)

[9] [Generalized Algebraic Difference Approach (GADA) Derivation](https://watermark.silverchair.com/forestscience0303.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA3cwggNzBgkqhkiG9w0BBwagggNkMIIDYAIBADCCA1kGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMzAFTzOheKIUFYmqnAgEQgIIDKhXA5UIuVNJSArI4KXQv5J_1UPuPpDFph9_MCov0B0C1BJo_u8RKo2NsHxwTGhaKQcW3e7oH5F_FWAzYPJQsJdh6Nykfk02GNPUh9Bp5OfqPd6Mj7mK5yDZAgKb0fx_yxd5mGucrIaEui0pjOtBdFbRiKuERRQO8vHJ-jEGRFIXx1ou17ukAiNkmsS97weFJMD2Y6vDEUAqBIpWdPUZ8jovbff6w7sIZ2XKlCDZ79j0inf9RgtpuuIraxRtcxQpRnrn7BK3KyXeCS8l198BOj-0SSy5FCvCtb7yIzI9JpWfoLaO7QpPxsV700bUxtrMUoNKsVzxk-DmuKWgGEiYgx6dizrKJWZ-LWx8iyRrzNz1R2RQe8pXwA0fDtAQrPL5HIcePd4zY0shmQkCwNm1c5hRL36gMtPVRDpxBA5dN6dwznEvHNBHyMt0cWN5VW5Mnwu9iIG9vEm5ikxNqF2HpaB2eyucEjqbRvXO-1qniCvYhPuFLqX1lzIVAN9SP4EF_lwennV23aJVs-WdaORF9ZCqjLBbX85bczjF0xoxnckNeXcazXmCoi0TDNH7xMZg5qbnQEHdJ-90I-B5IulCk6a-Sp7HA5QdP0akvGt_Nb5TKHUTynR3hHbfzgxF7fQsrnuNXZ4dUcbIB8mwWjIeLEWpLFcLfcSQDw6RXDfF5nYeoS4izjxYyld0n0GEcamnfOtijMoqKEh6gARmEQU-HlOJOhGsOCjKBLwyKIsBLYkYsb7dJdLFd7nUppEXctQUQWZZAUfF-5ob8XZmPO8Fa6JIJ0cTkbPSpCq_ehh6Harg3g0mSUMVtFJgNsZy9JygC91Nce2Wn9DOl_1YoGbyAnVhPTOETUlPcYyRvEX8m-QXETowImGBHDMH4ikb3xjpUaBL34uwi1UlbkY_pRe6t6V6jS5vSWqT-B6D8Mic-L1LaGpzAmvgITnveTBXPngMHM2OR4fqQC5WM-6NoGnEQ4dpUndS0z5yKSZ5avys6njNctcZcDWttzE37wjSph2TcHnvoh2pId6K7jcaa_ADyuM-pIkm7CMx08McaSqEYRDZnfwGe6yAMXkqVJA)

[10] [Models of dominant height growth and site indexes: GADA Derivation](https://www.scielo.org.mx/scielo.php?pid=S1405-31952018000300437&script=sci_arttext_plus&tlng=en)

[11] [Two-sample Kolmogorov-Smirnov Test in PySpark](https://github.com/Davi-Schumacher/KS-2Samp-PySparkSQL)
