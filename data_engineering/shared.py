import io
import os
import boto3
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import warnings
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import CCIIndicator, ADXIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

#! Surpressing just the ScalarCalcs of TA-lib caused by rolling calculations & Pandas Futurewarning
#! Pandas Futurewarning has a issue assigned to it for version 2.1 (https://github.com/dmlc/xgboost/issues/9543)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='ta')

load_dotenv()
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_SECRET')
S3_BUCKET = os.getenv('S3_BUCKET')
BASE_PRICE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
BASE_NEWS_URL = "https://api.polygon.io/v2/reference/news?ticker={ticker}&published_utc.lte={end_date}&order=asc&limit=1000&sort=published_utc&apiKey={api_key}"


class S3Handler:
    def __init__(self, aws_access_key, aws_secret_access_key, region='ap-northeast-2'):
        self.aws_access_key = aws_access_key
        self.aws_secret_access_key = aws_secret_access_key
        self.region = region
        self.session = None
        self.s3 = None

    def __enter__(self):
        self.session = boto3.Session(
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region
        )
        self.s3 = self.session.resource('s3')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def list_files(self, bucket, prefix):
        bucket = self.s3.Bucket(bucket)
        return [obj.key for obj in bucket.objects.filter(Prefix=prefix) if not obj.key.endswith('/')]

    def get_object_content(self, bucket, key):
        obj = self.s3.Object(bucket, key)
        return obj.get()['Body'].read()

def read_parquet_from_s3(prefix):
    full_prefix = f"xgb/lake/parquet/{prefix}/"
    with S3Handler(AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY) as s3_handler:
        files = s3_handler.list_files(S3_BUCKET, full_prefix)
        dataframes = []
        for file in files:
            if not file.endswith('.parquet'):
                continue
            content = s3_handler.get_object_content(S3_BUCKET, file)
            df = pd.read_parquet(io.BytesIO(content))
            dataframes.append(df)

        final_df = pd.concat(dataframes, ignore_index=True)
        return final_df

def aws_spark(app_name):
    builder = SparkSession.builder.appName(app_name)
    builder.config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0,org.apache.hadoop:hadoop-aws:3.2.2")
    builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    builder.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    builder.config("spark.sql.adaptive.enabled", "true")
    builder.config("spark.sql.execution.arrow.pyspark.enabled", "false")
    builder.config("spark.driver.memory", "4g")
    builder.config("spark.executor.memory", "4g")
    builder.config("spark.worker.memory", "4g")
    builder.config("spark.memory.fraction", "0.7")
    builder.config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY)
    builder.config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
    builder.config("spark.hadoop.fs.s3a.endpoint", "s3.ap-northeast-2.amazonaws.com")
    builder.config("spark.sql.sources.partitionOverwriteMode", "dynamic")
    builder.config("spark.databricks.delta.schema.autoMerge.enabled", "true")
    builder.config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    builder.config("spark.sql.execution.arrow.pyspark.selfDestruct.enabled", "true")
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    
    return spark


schema = StructType([
    StructField("t", TimestampType()),
    StructField("o", DoubleType()),
    StructField("h", DoubleType()),
    StructField("l", DoubleType()),
    StructField("c", DoubleType()),
    StructField("v", DoubleType()),
    StructField("vw", DoubleType()),
    StructField("n", LongType()),
    StructField("SPY_o", DoubleType()),
    StructField("SPY_h", DoubleType()),
    StructField("SPY_l", DoubleType()),
    StructField("SPY_c", DoubleType()),
    StructField("SPY_v", DoubleType()),
    StructField("SPY_vw", DoubleType()),
    StructField("SPY_n", LongType()),
    StructField("QQQ_o", DoubleType()),
    StructField("QQQ_h", DoubleType()),
    StructField("QQQ_l", DoubleType()),
    StructField("QQQ_c", DoubleType()),
    StructField("QQQ_v", DoubleType()),
    StructField("QQQ_vw", DoubleType()),
    StructField("QQQ_n", LongType()),
    StructField("DIA_o", DoubleType()),
    StructField("DIA_h", DoubleType()),
    StructField("DIA_l", DoubleType()),
    StructField("DIA_c", DoubleType()),
    StructField("DIA_v", DoubleType()),
    StructField("DIA_vw", DoubleType()),
    StructField("DIA_n", LongType()),
    StructField("IWM_o", DoubleType()),
    StructField("IWM_h", DoubleType()),
    StructField("IWM_l", DoubleType()),
    StructField("IWM_c", DoubleType()),
    StructField("IWM_v", DoubleType()),
    StructField("IWM_vw", DoubleType()),
    StructField("IWM_n", LongType()),
    StructField("VTI_o", DoubleType()),
    StructField("VTI_h", DoubleType()),
    StructField("VTI_l", DoubleType()),
    StructField("VTI_c", DoubleType()),
    StructField("VTI_v", DoubleType()),
    StructField("VTI_vw", DoubleType()),
    StructField("VTI_n", LongType()),
    StructField("EFA_o", DoubleType()),
    StructField("EFA_h", DoubleType()),
    StructField("EFA_l", DoubleType()),
    StructField("EFA_c", DoubleType()),
    StructField("EFA_v", DoubleType()),
    StructField("EFA_vw", DoubleType()),
    StructField("EFA_n", LongType()),
    StructField("BIL_o", DoubleType()),
    StructField("BIL_h", DoubleType()),
    StructField("BIL_l", DoubleType()),
    StructField("BIL_c", DoubleType()),
    StructField("BIL_v", DoubleType()),
    StructField("BIL_vw", DoubleType()),
    StructField("BIL_n", LongType()),
    StructField("SHV_o", DoubleType()),
    StructField("SHV_h", DoubleType()),
    StructField("SHV_l", DoubleType()),
    StructField("SHV_c", DoubleType()),
    StructField("SHV_v", DoubleType()),
    StructField("SHV_vw", DoubleType()),
    StructField("SHV_n", LongType()),
    StructField("IEI_o", DoubleType()),
    StructField("IEI_h", DoubleType()),
    StructField("IEI_l", DoubleType()),
    StructField("IEI_c", DoubleType()),
    StructField("IEI_v", DoubleType()),
    StructField("IEI_vw", DoubleType()),
    StructField("IEI_n", LongType()),
    StructField("IEF_o", DoubleType()),
    StructField("IEF_h", DoubleType()),
    StructField("IEF_l", DoubleType()),
    StructField("IEF_c", DoubleType()),
    StructField("IEF_v", DoubleType()),
    StructField("IEF_vw", DoubleType()),
    StructField("IEF_n", LongType()),
    StructField("TLT_o", DoubleType()),
    StructField("TLT_h", DoubleType()),
    StructField("TLT_l", DoubleType()),
    StructField("TLT_c", DoubleType()),
    StructField("TLT_v", DoubleType()),
    StructField("TLT_vw", DoubleType()),
    StructField("TLT_n", LongType()),
    StructField("next_c", DoubleType()),
    StructField("target_class", LongType()),
    StructField("c_class", LongType()),
    StructField("vw_lag_1", DoubleType()),
    StructField("vw_lag_2", DoubleType()),
    StructField("vw_lag_3", DoubleType()),
    StructField("vw_lag_4", DoubleType()),
    StructField("vw_lag_5", DoubleType()),
    StructField("n_lag_1", LongType()),
    StructField("n_lag_2", LongType()),
    StructField("n_lag_3", LongType()),
    StructField("n_lag_4", LongType()),
    StructField("n_lag_5", LongType()),
    StructField("vw_rolling_mean", DoubleType()),
    StructField("vw_rolling_std", DoubleType()),
    StructField("n_rolling_mean", DoubleType()),
    StructField("n_rolling_std", DoubleType()),
    StructField("aggregated_text", StringType()),
    StructField("target", DoubleType()),
    StructField("williams_r", DoubleType()),
    StructField("stoch_k", DoubleType()),
    StructField("roc", DoubleType()),
    StructField("cci", DoubleType()),
    StructField("adx", DoubleType()),
    StructField("atr", DoubleType()),
    StructField("20D-EMA", DoubleType()),
    StructField("50D-EMA", DoubleType()),
    StructField("100D-EMA", DoubleType()),
    StructField("bollinger_mavg", DoubleType()),
    StructField("bollinger_hband", DoubleType()),
    StructField("bollinger_lband", DoubleType()),
    StructField("vwap", DoubleType()),
    StructField("pvwap", DoubleType()),
    StructField("rsi", DoubleType()),
    StructField("rsicat", DoubleType()),
    StructField("dayofweek", DoubleType()),
    StructField("month", DoubleType()),
    StructField("quarter", DoubleType()),
    StructField("MACDs_12_26_9", DoubleType()),
    StructField("MACDh_12_26_9", DoubleType()),
    StructField("DMP_16", DoubleType()),
    StructField("grouping_key", IntegerType())
])

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def apply_technical_features(pdf):
    # The calculations
    williams_r = WilliamsRIndicator(high=pdf['h'], low=pdf['l'], close=pdf['c']).williams_r()
    stochastic_oscillator = StochasticOscillator(high=pdf['h'], low=pdf['l'], close=pdf['c'])
    roc = ROCIndicator(pdf['c']).roc()
    cci = CCIIndicator(high=pdf['h'], low=pdf['l'], close=pdf['c']).cci()
    adx_indicator = ADXIndicator(high=pdf['h'], low=pdf['l'], close=pdf['c'])
    atr = AverageTrueRange(high=pdf['h'], low=pdf['l'], close=pdf['c']).average_true_range()
    ema20 = EMAIndicator(close=pdf['c'], window=20).ema_indicator()
    ema50 = EMAIndicator(close=pdf['c'], window=50).ema_indicator()
    ema100 = EMAIndicator(close=pdf['c'], window=100).ema_indicator()
    bollinger = BollingerBands(close=pdf['c'])
    vwap = np.cumsum(pdf['v'] * (pdf['h'] + pdf['l'] + pdf['c']) / 3) / np.cumsum(pdf['v'])
    rsi = RSIIndicator(close=pdf['c']).rsi()
    macd = MACD(close=pdf['c'])
    dmp_16 = adx_indicator.adx_pos()

    # Transformations(Calculations)
    pdf['williams_r'] = williams_r
    pdf['stoch_k'] = stochastic_oscillator.stoch()
    pdf['roc'] = roc
    pdf['cci'] = cci
    pdf['adx'] = adx_indicator.adx()
    pdf['atr'] = atr
    pdf['20D-EMA'] = ema20
    pdf['50D-EMA'] = ema50
    pdf['100D-EMA'] = ema100
    pdf['bollinger_mavg'] = bollinger.bollinger_mavg()
    pdf['bollinger_hband'] = bollinger.bollinger_hband()
    pdf['bollinger_lband'] = bollinger.bollinger_lband()
    pdf['vwap'] = vwap
    pdf['pvwap'] = pdf['c'] - vwap
    pdf['rsi'] = rsi
    pdf['rsicat'] = pd.cut(rsi, bins=[0, 30, 70, 100], labels=[1, 2, 3], right=False).astype('double') 
    pdf['t'] = pd.to_datetime(pdf['t'])
    pdf['dayofweek'] = pdf['t'].dt.dayofweek + 1
    pdf['month'] = pdf['t'].dt.month
    pdf['quarter'] = pdf['t'].dt.quarter
    pdf['MACDs_12_26_9'] = macd.macd_signal()
    pdf['MACDh_12_26_9'] = macd.macd_diff()
    pdf['DMP_16'] = dmp_16
    
    pdf = pdf.bfill()

    return pdf


