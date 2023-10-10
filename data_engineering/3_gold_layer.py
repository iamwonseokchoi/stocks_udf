
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from shared import aws_spark, apply_technical_features
from concurrent.futures import ThreadPoolExecutor
import pyspark.sql.functions as F
from dotenv import load_dotenv
import pyspark.sql.types as T
import warnings
import logging
import json
import os


#! Surpressing just the ScalarCalcs of TA-lib caused by rolling calculations & Pandas Futurewarning
#! Pandas Futurewarning has a issue assigned to it for version 2.1 (https://github.com/dmlc/xgboost/issues/9543)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='ta')


# Initial setup
load_dotenv()
s3_base_path = os.getenv('S3_BASE_PATH')
tickers = json.loads(os.getenv('TICKERS'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

spark = aws_spark("GoldLayer")
silver_price_path = s3_base_path + "/xgb/lake/price/silver/{}"


def dense_vector_to_array(v):
    return v.toArray().tolist()

def generate_gold_layer(ticker):
    logging.info(f"Starting Gold Layer Generation for {ticker}")
    
    df = spark.read.format("delta").load(silver_price_path.format(ticker)) \
        .withColumn('t', F.col('t').cast('String'))
    
    # Add in Grouping Key for Vectorized UDF (Simply just made 1 as the dataset is not large)
    grouping_df = df.withColumn('grouping_key', F.lit(1))
    
    logging.info(f"Starting Vectorized UDF Calculations for {ticker}")
    # Apply Vectorized UDF Calculations
    udf_df = grouping_df.groupBy('grouping_key').apply(apply_technical_features)

    logging.info(f"Done UDF Calculations for {ticker}.. Starting Text Processing for {ticker}")
    # Tokenization 
    tokenizer = Tokenizer(inputCol="aggregated_text", outputCol="toeknized_text")
    tokenized_DF = tokenizer.transform(udf_df)

    # Stopwords Removal
    remover = StopWordsRemover(inputCol="toeknized_text", outputCol="filtered_text")
    stopwords_DF = remover.transform(tokenized_DF)

    # Embedding (Word2Vec)
    word2Vec_descriptions = Word2Vec(vectorSize=100, minCount=0, inputCol="filtered_text", outputCol="embedded_text")
    model = word2Vec_descriptions.fit(stopwords_DF)
    embedded_DF = model.transform(stopwords_DF)
    
    logging.info(f"Done Text Processing for {ticker}.. Starting Final Pre-processing for {ticker}")
    
    embedded_DF = embedded_DF.drop('aggregated_text', 'toeknized_text', 'filtered_text')
    embeddedDF = embedded_DF.withColumn('embedded_text', vector_to_array_udf('embedded_text'))

    # Final Ordering
    final_cols_order = [col for col in embeddedDF.columns if col not in ['grouping_key', 'target']]
    final_cols_order.append('target')
    
    final_DF = embeddedDF.select(*final_cols_order)
    
    # To S3
    logging.info(f"Done Final Pre-processing for {ticker}.. Saving to S3 for {ticker}")    
    final_DF.write.format("delta") \
        .mode("overwrite") \
        .save(f"{s3_base_path}/xgb/lake/price/gold/{ticker}")
    
    final_DF.write.mode('overwrite').parquet(f"{s3_base_path}/xgb/lake/parquet/{ticker}")
    
    logging.info(f"** Done Gold Layer Generation for {ticker}")

if __name__ == "__main__":
    vector_to_array_udf = F.udf(dense_vector_to_array, T.ArrayType(T.FloatType()))

    with ThreadPoolExecutor() as executor:
        executor.map(generate_gold_layer, tickers)
    
    spark.stop()
    logging.info("====Done Gold Layer Generation====")
