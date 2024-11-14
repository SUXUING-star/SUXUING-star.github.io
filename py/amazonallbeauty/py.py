import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from collections import Counter
import glob
import os

def load_and_merge_data():
    """Load and merge multiple JSON files into a single DataFrame using pattern matching."""
    all_data = []
    # 使用glob模块查找所有匹配的文件
    json_files = glob.glob('E:\\pyt\\amazon reviews\\amazonallbeauty\\amazon-Products*.json')
    
    if not json_files:
        raise FileNotFoundError("No matching JSON files found!")
    
    print(f"Found {len(json_files)} JSON files:")
    for file in json_files:
        print(f"Loading: {file}")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
            print(f"Successfully loaded {len(data)} records from {file}")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    if not all_data:
        raise ValueError("No data was loaded from the files!")
    
    print(f"\nTotal records loaded: {len(all_data)}")
    return pd.DataFrame(all_data)

def clean_price(price):
    """Convert price string to float."""
    if pd.isna(price) or price == '':
        return np.nan
    return float(re.sub(r'[^\d.]', '', price))

def clean_rating(rating):
    """Convert rating string to float."""
    if pd.isna(rating):
        return np.nan
    match = re.search(r'(\d+\.?\d*)', str(rating))
    return float(match.group(1)) if match else np.nan

def clean_ratingnum(ratingnum):
    """Convert rating number string to integer."""
    if pd.isna(ratingnum) or ratingnum == '':
        return 0
    return int(re.sub(r'[^\d]', '', str(ratingnum)))

def clean_data(df):
    """Clean and standardize the DataFrame."""
    print("\nCleaning data...")
    initial_rows = len(df)
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Clean price
    df_clean['price_clean'] = df_clean['price'].apply(clean_price)
    
    # Clean rating
    df_clean['rating_clean'] = df_clean['rating'].apply(clean_rating)
    
    # Clean rating numbers
    df_clean['ratingnum_clean'] = df_clean['ratingnum'].apply(clean_ratingnum)
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['title'])
    
    print(f"Initial number of records: {initial_rows}")
    print(f"Records after cleaning: {len(df_clean)}")
    print(f"Removed {initial_rows - len(df_clean)} duplicate records")
    
    # Print basic data quality statistics
    print("\nData quality check:")
    print(f"Missing prices: {df_clean['price_clean'].isna().sum()}")
    print(f"Missing ratings: {df_clean['rating_clean'].isna().sum()}")
    
    return df_clean

def create_visualizations(df):
    """Create and save visualizations."""
    print("\nCreating visualizations...")
    
    # Create output directory if it doesn't exist
    output_dir = 'beauty_analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    
    # 1. Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='price_clean', bins=30)
    plt.title('Price Distribution of Beauty Products')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/price_distribution.png')
    plt.close()
    
    # 2. Rating Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='rating_clean', bins=20)
    plt.title('Rating Distribution of Beauty Products')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/rating_distribution.png')
    plt.close()
    
    # 3. Price vs Rating Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='price_clean', y='rating_clean', 
                    size='ratingnum_clean', sizes=(20, 200),
                    alpha=0.6)
    plt.title('Price vs Rating (size indicates number of ratings)')
    plt.xlabel('Price ($)')
    plt.ylabel('Rating')
    plt.savefig(f'{output_dir}/price_vs_rating.png')
    plt.close()
    
    # 4. Word Cloud
    text = ' '.join(df['title'].dropna())
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         max_words=100).generate(text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Product Titles')
    plt.savefig(f'{output_dir}/wordcloud.png')
    plt.close()
    
    print(f"Visualizations saved in '{output_dir}' directory")

def generate_summary_stats(df):
    """Generate summary statistics."""
    stats = {
        'Total Products': len(df),
        'Average Price': f"${df['price_clean'].mean():.2f}",
        'Median Price': f"${df['price_clean'].median():.2f}",
        'Price Range': f"${df['price_clean'].min():.2f} - ${df['price_clean'].max():.2f}",
        'Average Rating': f"{df['rating_clean'].mean():.2f}",
        'Most Common Rating': f"{df['rating_clean'].mode().iloc[0]:.1f}",
        'Total Reviews': df['ratingnum_clean'].sum(),
    }
    return stats

def main():
    try:
        # Load and merge data
        df = load_and_merge_data()
        
        # Clean data
        df_clean = clean_data(df)
        
        # Create visualizations
        create_visualizations(df_clean)
        
        # Generate and print summary statistics
        stats = generate_summary_stats(df_clean)
        print("\nSummary Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Save cleaned data
        output_dir = 'beauty_analysis_output'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}/cleaned_beauty_products.csv'
        df_clean.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()