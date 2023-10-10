from pyspark.sql.window import Window
import pyspark.sql.functions as F
from dotenv import load_dotenv
from shared import aws_spark
import logging
import json
import os
import threading


# Initial setup
load_dotenv()
s3_base_path = os.getenv('S3_BASE_PATH')
tickers = json.loads(os.getenv('TICKERS'))
proxies = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'EFA', 'BIL', 'SHV', 'IEI', 'IEF', 'TLT']

spark = aws_spark("SilverLayer")
bronze_price_path = f"{s3_base_path}/xgb/lake/price/bronze"
bronze_news_path = f"{s3_base_path}/xgb/lake/news/bronze"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def backfill_nulls(df, order_by_column_name):
    windowSpec = Window.orderBy(order_by_column_name).rowsBetween(Window.unboundedPreceding, 0)
    for column_name in df.columns:
        backfilled_column = F.last(df[column_name], ignorenulls=True).over(windowSpec)
        df = df.withColumn(column_name, F.when(F.col(column_name).isNotNull(), F.col(column_name)).otherwise(backfilled_column))
    
    return df

def fill_initial_nulls(df, order_by_column_name):
    windowSpec = Window.orderBy(order_by_column_name).rowsBetween(0, Window.unboundedFollowing)
    for column_name in df.columns:
        nearest_value_column = F.first(df[column_name], ignorenulls=True).over(windowSpec)
        df = df.withColumn(column_name, F.when(F.col(column_name).isNotNull(), F.col(column_name)).otherwise(nearest_value_column))
    
    return df

def process_news_for_ticker(ticker):
    df_news = spark.read.format('delta') \
        .load(bronze_news_path) \
        .filter(F.col('search_ticker') == ticker) \
        .select("published_utc", "title", "description", "keywords")
    
    df_news = df_news.withColumn('published_utc', F.date_trunc('day', F.col('published_utc')))
    df_news = df_news.withColumn("combined_text", 
                    F.concat_ws(" ", 
                                df_news.description, 
                                F.concat_ws(" ", df_news.keywords), 
                                df_news.title))
    
    df_news = (
        df_news.groupBy("published_utc")
        .agg(F.concat_ws(" ", F.collect_list("combined_text")).alias("aggregated_text"))
    )

    return df_news.withColumn("aggregated_text", F.regexp_replace(df_news.aggregated_text, "\n", " "))

def process_data_for_ticker(ticker, proxies, df_news):
    df = spark.read.format("delta").load(bronze_price_path).filter(F.col('ticker') == ticker)
    for proxy in proxies:
        df_proxy = spark.read.format('delta') \
            .load(bronze_price_path).filter(F.col('ticker') == proxy) \
            .select('t', 'o', 'h', 'l', 'c', 'v', 'vw', 'n') \
            .withColumnRenamed('o', f'{proxy}_o') \
            .withColumnRenamed('h', f'{proxy}_h') \
            .withColumnRenamed('l', f'{proxy}_l') \
            .withColumnRenamed('c', f'{proxy}_c') \
            .withColumnRenamed('v', f'{proxy}_v') \
            .withColumnRenamed('vw', f'{proxy}_vw') \
            .withColumnRenamed('n', f'{proxy}_n')

        df = df.join(df_proxy, on='t', how='left')

    df = backfill_nulls(df, 't')
    df = fill_initial_nulls(df, 't')
    df = df.withColumn("t", F.date_format(F.from_unixtime(F.col("t") / 1000), "yyyy-MM-dd HH:mm:ss")) \
            .withColumn("t", F.to_timestamp(F.col("t"), "yyyy-MM-dd HH:mm:ss")) \
            .withColumn('t', F.date_trunc('day', F.col('t')))
    
    result_df = (df.join(df_news, df.t == df_news.published_utc, how="left")
                    .drop("published_utc"))

    return result_df.withColumn("aggregated_text", F.coalesce(result_df.aggregated_text, F.lit(" ")))

def process_ticker(ticker, proxies):
    logging.info(f"Starting processing for {ticker}")

    # Process news for the ticker
    df_news = process_news_for_ticker(ticker)

    # Process data for the ticker
    DF_target = process_data_for_ticker(ticker, proxies, df_news)

    # Creating target for the ticker
    window_spec = Window.orderBy('t')
    DF_target = DF_target \
        .withColumn('target', F.col('c') - F.col('o')) \
        .withColumn('target', F.lead('target').over(window_spec))
    DF_target = DF_target \
        .withColumn('next_c', F.lead('c').over(window_spec)) \
        .withColumn('target_class', F.when(F.col('target') > 0, 1).otherwise(0)) \
        .withColumn('c_class', F.when(F.col('c') > F.col('o'), 1).otherwise(0))

    logging.info(f"Creating lag features for {ticker}")

    lookback = 5
    cols_to_lag = ['vw', 'n']
    for col in cols_to_lag:
        for lag in range(1, lookback+1):
            DF_target = DF_target.withColumn(f'{col}_lag_{lag}', F.lag(DF_target[col], lag).over(window_spec))

    logging.info(f"Creating rolling features for {ticker}")

    rolling_spec = Window.orderBy('t').rowsBetween(-(lookback), -1)
    for col in cols_to_lag:
        DF_target = DF_target.withColumn(f'{col}_rolling_mean', F.avg(DF_target[col]).over(rolling_spec)) \
            .withColumn(f'{col}_rolling_std', F.stddev(DF_target[col]).over(rolling_spec))

    DF_target = DF_target.dropna()

    logging.info(f"Final processing for {ticker}")

    cols = [col for col in DF_target.columns if col not in ['target', 'ticker']]
    cols = cols + ['target']

    DF_target = DF_target.select(*cols)

    logging.info(f"Saving {ticker} to S3")

    DF_target.write.format('delta') \
        .mode('overwrite') \
        .save(f"{s3_base_path}/xgb/lake/price/silver/{ticker}")

    logging.info(f"** Done {ticker} to S3")



if __name__ == "__main__":
    logging.info("Starting Silver Layer")
    
    threads = []
    for ticker in tickers:
        t = threading.Thread(target=process_ticker, args=(ticker, proxies))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    spark.stop()
    logging.info("All tickers processed and Silver Layer completed.")

