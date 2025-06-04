from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace
from pyspark.sql.types import FloatType

def process_silver(spark, bronze_df):
    silver_df = bronze_df.dropDuplicates()
    columns_to_drop = ['price', 'host_review_score']
    silver_df = silver_df.drop(*columns_to_drop)

    silver_df = silver_df.withColumn('Giá trước',
        when(col('Giá trước') == 'Không có giá trước', None).otherwise(
            regexp_replace(col('Giá trước'), r'\.', '').cast(FloatType())
        )
    ).withColumn('Giá sau',
        when(col('Giá sau') == 'Không có giá sau', None).otherwise(
            regexp_replace(col('Giá sau'), r'\.', '').cast(FloatType())
        )
    )

    silver_df = silver_df.withColumn('full',
        when((col('Giá trước').isNull()) & (col('Giá sau').isNull()), 1).otherwise(0)
    )
    silver_df = silver_df.withColumn('Giá trước',
        when(col('Giá trước').isNull(), col('Giá sau')).otherwise(col('Giá trước'))
    ).withColumn('Giá trước',
        when(col('Giá trước').isNull(), 0).otherwise(col('Giá trước'))
    ).withColumn('Giá sau',
        when(col('Giá sau').isNull(), 0).otherwise(col('Giá sau'))
    )

    silver_df = silver_df.withColumn('review_score',
        regexp_replace(col('review_score'), r',', '.').cast(FloatType())
    )

    cols_to_fill = ['review_score', 'review_count', 'Vị trí', 'Dịch vụ', 'Đáng giá tiền', 'Cơ sở vật chất', 'Độ sạch sẽ']
    for col_name in cols_to_fill:
        silver_df = silver_df.withColumn(col_name,
            when(col(col_name).cast('string').rlike(r'^Không có.*'), None)
            .otherwise(regexp_replace(col(col_name).cast('string'), r',', '.').cast(FloatType()))
        )

    fill_values = {col_name: 5.0 for col_name in ['review_score', 'Vị trí', 'Dịch vụ', 'Đáng giá tiền', 'Cơ sở vật chất', 'Độ sạch sẽ']}
    fill_values['review_count'] = 0.0
    silver_df = silver_df.fillna(fill_values)

    silver_df = silver_df.withColumnRenamed('name', 'ten')\
        .withColumnRenamed('address', 'dia_chi')\
        .withColumnRenamed('review_score', 'diem_danh_gia')\
        .withColumnRenamed('review_count', 'so_luong_danh_gia')\
        .withColumnRenamed('Vị trí', 'vi_tri')\
        .withColumnRenamed('Dịch vụ', 'dich_vu')\
        .withColumnRenamed('Đáng giá tiền', 'dang_gia_tien')\
        .withColumnRenamed('Cơ sở vật chất', 'co_so_vat_chat')\
        .withColumnRenamed('Độ sạch sẽ', 'do_sach_se')\
        .withColumnRenamed('Giá trước', 'gia_truoc')\
        .withColumnRenamed('Giá sau', 'gia_sau')\
        .withColumnRenamed('location', 'vi_tri_map')\
        .withColumnRenamed('date', 'ngay')\
        .withColumnRenamed('full', 'het_phong')

    silver_table_path = os.path.abspath("delta/silver/hotels")
    os.makedirs(silver_table_path, exist_ok=True)
    silver_df.write.mode("overwrite").format("delta").save(silver_table_path)
    spark.sql(f"CREATE TABLE IF NOT EXISTS silver_hotels USING DELTA LOCATION '{silver_table_path}'")
    return silver_df