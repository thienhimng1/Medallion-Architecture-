from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, lit
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression

def process_gold(spark, silver_df):
    gold_df = silver_df.drop('gia_truoc')
    hotel_df = gold_df.dropDuplicates(['ten'])
    mean_score = hotel_df.select(mean(col('diem_danh_gia')).alias('mean_score')).collect()[0]['mean_score']
    m = hotel_df.approxQuantile('so_luong_danh_gia', [0.75], 0.05)[0]

    hotel_df = hotel_df.withColumn('gia_tri_thuc_netflix',
        (col('so_luong_danh_gia') / (col('so_luong_danh_gia') + m) * col('diem_danh_gia') +
         m / (col('so_luong_danh_gia') + m) * mean_score).cast('float')
    )
    gold_df = gold_df.join(hotel_df.select('ten', 'gia_tri_thuc_netflix'), on='ten', how='left')

    features = ['bep', 'bua_sang', 'san_gon', 'may_giat', 'dua_don_san_bay',
                'phong_tap', 'don_phong_hang_ngay', 'thu_nuoi', 'bai_do_xe']
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    gold_df = assembler.transform(gold_df)
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(gold_df)
    gold_df = scaler_model.transform(gold_df)

    yeu_thich = ['KHÁCH SẠN LA VELA SÀI GÒN (LA VELA SAIGON HOTEL)', 'Hotel Continental Saigon', 'COCHIN SANG HOTEL']
    khong_thich = ['Khách sạn Minh Anh Gò Vấp (Khách sạn Minh Anh Gò Vấp )', 'Khách Sạn Mari Queen (Mari Queen Hotel)']
    gold_df = gold_df.withColumn('target',
        when(col('ten').isin(yeu_thich), 1)
        .when(col('ten').isin(khong_thich), -1)
        .otherwise(0)
    )

    kmeans = KMeans(featuresCol='scaled_features', k=4, seed=42)
    kmeans_model = kmeans.fit(gold_df)
    gold_df = kmeans_model.transform(gold_df).withColumnRenamed('prediction', 'cluster')

    def evaluate_cluster(df):
        from pyspark.sql.functions import sum as sum_
        result = df.groupBy('cluster').agg(
            sum_(when(col('target') == 1, 1).otherwise(0)).alias('positive_count'),
            sum_(when(col('target') == -1, 1).otherwise(0)).alias('negative_count')
        )
        result = result.withColumn('cum_danh_gia',
            when(col('positive_count') > col('negative_count'), 'GẦN GU THÍCH')
            .when(col('negative_count') > col('positive_count'), 'KHÔNG HỢP GU')
            .otherwise('TRUNG LẬP')
        )
        return result.select('cluster', 'cum_danh_gia')

    cluster_eval = evaluate_cluster(gold_df)
    gold_df = gold_df.join(cluster_eval, on='cluster', how='left')

    gold_df = gold_df.withColumn('target_khuech_dai',
        when(col('cum_danh_gia') == 'GẦN GU THÍCH', 1)
        .when(col('cum_danh_gia') == 'KHÔNG HỢP GU', -1)
        .otherwise(col('target'))
    )

    lr = LinearRegression(featuresCol='scaled_features', labelCol='target_khuech_dai', regParam=1.0)
    lr_model = lr.fit(gold_df)
    gold_df = lr_model.transform(gold_df).withColumnRenamed('prediction', 'gia_tri_thuc_ridge')

    assembler = VectorAssembler(inputCols=['gia_tri_thuc_ridge'], outputCol='ridge_vec')
    gold_df = assembler.transform(gold_df)
    scaler = MinMaxScaler(inputCol='ridge_vec', outputCol='ridge_scaled_vec', min=0, max=10)
    scaler_model = scaler.fit(gold_df)
    gold_df = scaler_model.transform(gold_df)
    gold_df = gold_df.withColumn('gia_tri_thuc_ridge_scaled', col('ridge_scaled_vec')[0])

    gold_df = gold_df.withColumn('gia_tri_thuc',
        (0.5 * col('gia_tri_thuc_ridge') + 0.5 * col('gia_tri_thuc_netflix')).cast('float')
    )
    assembler = VectorAssembler(inputCols=['gia_tri_thuc'], outputCol='gia_tri_thuc_vec')
    gold_df = assembler.transform(gold_df)
    scaler = MinMaxScaler(inputCol='gia_tri_thuc_vec', outputCol='gia_tri_thuc_scaled_vec', min=0, max=10)
    scaler_model = scaler.fit(gold_df)
    gold_df = scaler_model.transform(gold_df)
    gold_df = gold_df.withColumn('gia_tri_thuc', col('gia_tri_thuc_scaled_vec')[0].cast('float'))

    columns_to_drop = ['target', 'target_khuech_dai', 'cum_danh_gia', 'cluster', 'gia_tri_thuc_ridge',
                       'gia_tri_thuc_ridge_scaled', 'gia_tri_thuc_netflix', 'features',
                       'scaled_features', 'ridge_vec', 'gia_tri_thuc_vec', 'gia_tri_thuc_scaled_vec']
    gold_df = gold_df.drop(*columns_to_drop)

    gold_table_path = os.path.abspath("delta/gold/hotels")
    os.makedirs(gold_table_path, exist_ok=True)
    gold_df.write.mode("overwrite").format("delta").save(gold_table_path)
    spark.sql(f"CREATE TABLE IF NOT EXISTS gold_hotels USING DELTA LOCATION '{gold_table_path}'")
    return gold_df