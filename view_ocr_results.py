"""
View OCR results with HTML stripped to plain text.
"""

import pandas as pd
from bs4 import BeautifulSoup


def extract_text_from_html(html_content: str) -> str:
    """Extract plain text from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator='\n', strip=True)
    return text


def main():
    # Load results
    df = pd.read_parquet('data/periodical_dataframes/post_processed/custom_newspaper.parquet')

    print(f"Total regions: {len(df)}")
    print(f"Columns: {df.columns.tolist()}\n")
    print("="*60)
    print("EXTRACTED TEXT:")
    print("="*60)

    # Extract and display plain text
    for idx, row in df.iterrows():
        text = extract_text_from_html(row['content'])
        print(f"\n--- Region {idx + 1} (Reading Order: {row['class']}) ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
