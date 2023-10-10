from pyspark.sql.types import StructType, StructField, StringType, FloatType, LongType
from datetime import date
import aiohttp
import asyncio
import logging
import json 
import os
from concurrent.futures import ThreadPoolExecutor

from shared import aws_spark, BASE_PRICE_URL, BASE_NEWS_URL
from dotenv import load_dotenv

load_dotenv()
s3_base_path = os.getenv('S3_BASE_PATH')
polygon_key = os.getenv('POLYGON_API_KEY')
tickers = json.loads(os.getenv('TICKERS'))
final_tickers = tickers.copy()
final_tickers.extend(['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'EFA', 'BIL', 'SHV', 'IEI', 'IEF', 'TLT'])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def fetch_data(session, url):
    async with session.get(url) as response:
        
        return await response.json()

async def get_ticker_data(session, ticker):
    end_date = date.today()
    start_date = "2019-01-01"  #! Adjust as needed
    url = BASE_PRICE_URL.format(ticker=ticker, start_date=start_date, end_date=end_date, api_key=polygon_key)
    all_data = []

    next_url = url
    while next_url:
        response_data = await fetch_data(session, next_url)
        all_data.extend(response_data.get('results', []))

        next_url = response_data.get('next_url')
        if next_url:
            next_url = f"{next_url}&apiKey={polygon_key}"

    return {ticker: all_data}

async def get_news_data(session, ticker):
    end_date = date.today()
    url = BASE_NEWS_URL.format(ticker=ticker, end_date=end_date, api_key=polygon_key)
    all_data = []

    next_url = url
    while next_url:
        response_data = await fetch_data(session, next_url)
        all_data.extend(response_data.get('results', []))

        next_url = response_data.get('next_url')
        if next_url:
            next_url = f"{next_url}&apiKey={polygon_key}"

    return {ticker: all_data}

async def call_price(tickers):
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(get_ticker_data(session, ticker) for ticker in final_tickers))
    return results

async def call_news(tickers):
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(get_news_data(session, ticker) for ticker in tickers))
    return results

def process_price_data(price_data):
    price_rows = []
    for ticker_data in price_data:
        for ticker, results in ticker_data.items():
            for result in results:
                price_rows.append(
                    (ticker, result["t"], float(result["o"]), float(result["h"]), float(result["l"]), float(result["c"]), 
                    float(result["v"]), float(result.get("vw", 0)), (result.get("n", 0))))
    return price_rows

def process_news_data(news_data):
    return [{"search_ticker": ticker, **data} for ticker_data in news_data for ticker, datas in ticker_data.items() for data in datas]

def run_in_thread(fn):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    with ThreadPoolExecutor() as pool:
        return pool.submit(fn).result()

if __name__ == "__main__":
    price_schema = StructType([
        StructField("ticker", StringType(), True),
        StructField("t", LongType(), True),
        StructField("o", FloatType(), True),
        StructField("h", FloatType(), True),
        StructField("l", FloatType(), True),
        StructField("c", FloatType(), True),
        StructField("v", FloatType(), True),
        StructField("vw", FloatType(), True),
        StructField("n", LongType(), True)
    ])
    
    logging.info("Starting Bronze Layer.. Calling Polygon API")
    
    price_data = run_in_thread(lambda: asyncio.run(call_price(tickers)))
    
    try: 
        news_data = run_in_thread(lambda: asyncio.run(call_news(tickers)))
    except Exception as e:
        logging.error(f"Error calling news API: {e}")
        pass
        
    logging.info("Done calling Polygon API.. Processing data")
    
    spark = aws_spark("BronzeLayer")

    # Process and Write Price to S3
    logging.info("Writing Price Data to S3")
    price_rows = process_price_data(price_data)
    price_df = spark.createDataFrame(price_rows, schema=price_schema)
    price_df.write.format("delta") \
        .mode("overwrite") \
        .partitionBy("ticker") \
        .save(f"{s3_base_path}/xgb/lake/price/bronze")

    try: 
        # Process and Write News to S3
        logging.info("Writing News Data to S3")
        news_rows = process_news_data(news_data)
        news_df = spark.createDataFrame(news_rows)
        news_df.write.format("delta") \
            .mode("overwrite") \
            .partitionBy("search_ticker") \
            .save(f"{s3_base_path}/xgb/lake/news/bronze")
    except Exception as e:
        logging.error(f"Error processing news data: {e}")
        pass

    spark.stop()