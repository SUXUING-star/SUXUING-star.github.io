import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datasets import load_dataset
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import gc

class AmazonReviewsAnalyzer:
    def __init__(self):
        """初始化分析器，加载完整数据集"""
        print("Initializing analyzer...")
        print("Loading full dataset...")
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
        self.data = pd.DataFrame(dataset["full"])
        print(f"Loaded {len(self.data)} reviews.")
        
        # 转换时间戳
        print("Converting timestamps...")
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], unit="ms")
        
        # 初始化基本的文本长度计算
        print("Calculating review lengths...")
        self.data['review_length'] = self.data['text'].apply(lambda x: len(str(x).split()))
        
    def get_basic_stats(self):
        """获取基本统计信息"""
        try:
            numeric_stats = {
                'Total_Reviews': len(self.data),
                'Average_Rating': float(self.data['rating'].mean()),
                'Verified_Purchases_Pct': float(self.data['verified_purchase'].mean() * 100),
                'Average_Review_Length': float(self.data['review_length'].mean()),
                'Five_Stars_Pct': float((self.data['rating'] == 5).mean() * 100),
                'Four_Stars_Pct': float((self.data['rating'] == 4).mean() * 100),
                'Three_Stars_Pct': float((self.data['rating'] == 3).mean() * 100),
                'Two_Stars_Pct': float((self.data['rating'] == 2).mean() * 100),
                'One_Star_Pct': float((self.data['rating'] == 1).mean() * 100)
            }
            
            numeric_results = pd.Series(numeric_stats).round(2)
            date_range = f"{self.data['timestamp'].min().date()} to {self.data['timestamp'].max().date()}"
            full_results = pd.Series({
                **numeric_results.to_dict(),
                'Date_Range': date_range
            })
            
            return full_results
        
        except Exception as e:
            print(f"Error in get_basic_stats: {str(e)}")
            print("Data types of columns:")
            print(self.data.dtypes)
            return None
    
    def preprocess_data(self, batch_size=10000):
        """数据预处理，使用批处理来优化内存使用"""
        print("Starting data preprocessing...")
        
        try:
            self.data['sentiment_score'] = np.nan
            
            for i in tqdm(range(0, len(self.data), batch_size), desc="Calculating sentiment scores"):
                batch = self.data.iloc[i:i+batch_size]
                self.data.loc[batch.index, 'sentiment_score'] = batch['text'].apply(
                    lambda x: TextBlob(str(x)).sentiment.polarity)
                gc.collect()
            
            # 检查是否存在价格列，并处理空值
            if 'price' in self.data.columns and not self.data['price'].isnull().all():
                print("Creating price ranges...")
                self.data['price_range'] = pd.qcut(
                    self.data['price'].fillna(0), 
                    q=5, 
                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
                )
            else:
                print("Warning: 'price' column is missing or has all NaN values.")
                self.data['price_range'] = 'Unknown'
            
        except Exception as e:
            print(f"Error in preprocess_data: {str(e)}")
            
    def analyze_ratings(self):
        """评分详细分析"""
        try:
            print("Analyzing ratings...")
            plt.figure(figsize=(18, 6))
            
            plt.subplot(131)
            sns.histplot(data=self.data, x="rating", bins=5, kde=True)
            plt.title("Rating Distribution")
            
            plt.subplot(132)
            sns.boxplot(data=self.data, x="verified_purchase", y="rating")
            plt.title("Ratings by Verified Purchase")
            
            plt.subplot(133)
            sns.boxplot(data=self.data, x="price_range", y="rating")
            plt.xticks(rotation=45)
            plt.title("Ratings by Price Range")
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in analyze_ratings: {str(e)}")

def main():
    try:
        analyzer = AmazonReviewsAnalyzer()
        
        print("\nBasic Statistics:")
        stats = analyzer.get_basic_stats()
        if stats is not None:
            print(stats)
            
        analyzer.preprocess_data()
        analyzer.analyze_ratings()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
