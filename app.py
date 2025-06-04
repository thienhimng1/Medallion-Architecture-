import streamlit as st
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from bronze_processing import process_bronze
from silver_processing import process_silver
from gold_processing import process_gold
import os
import pandas as pd

# Hàm khởi tạo Spark
def init_spark():
    builder = SparkSession.builder \
        .appName("HotelDataProcessing") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.warehouse.dir", "spark-warehouse") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g")
    return configure_spark_with_delta_pip(builder).getOrCreate()

# Streamlit app
st.title("Ứng dụng Xử lý và Gợi ý Khách sạn")
st.write("Ứng dụng này thực hiện xử lý dữ liệu khách sạn qua các tầng Bronze, Silver, Gold và hiển thị kết quả.")

# Đường dẫn dữ liệu
folder_path = "RawData"  # Thư mục RawData trong repository

# Kiểm tra thư mục tồn tại
if not os.path.exists(folder_path):
    st.error("Thư mục RawData không tồn tại. Vui lòng tạo thư mục và đặt các file CSV vào đó.")
else:
    spark = init_spark()

    # Nút xử lý dữ liệu
    if st.button("Xử lý dữ liệu"):
        try:
            with st.spinner("Đang xử lý tầng Bronze..."):
                bronze_df = process_bronze(spark, folder_path)
                st.success(f"Đã xử lý {bronze_df.count()} bản ghi trong tầng Bronze")

            with st.spinner("Đang xử lý tầng Silver..."):
                silver_df = process_silver(spark, bronze_df)
                st.success(f"Đã xử lý {silver_df.count()} bản ghi trong tầng Silver")

            with st.spinner("Đang xử lý tầng Gold..."):
                gold_df = process_gold(spark, silver_df)
                st.success(f"Đã xử lý {gold_df.count()} bản ghi trong tầng Gold")

            # Hiển thị kết quả
            st.subheader("Kết quả từ tầng Gold")
            gold_pd = gold_df.select('ten', 'gia_sau', 'diem_danh_gia', 'so_luong_danh_gia', 'gia_tri_thuc').toPandas()
            st.dataframe(gold_pd)

            # Vẽ biểu đồ giá trị thực
            st.subheader("Biểu đồ Giá trị Thực")
            st.bar_chart(gold_pd.set_index('ten')['gia_tri_thuc'])

        except Exception as e:
            st.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")

    spark.stop()
