"""
arXiv Paper Fetcher

A comprehensive tool for fetching and categorizing the latest Computer Science papers from arXiv.
Provides intelligent paper categorization, GitHub URL extraction, and optional PDF downloading.

Author: AGI Lab
License: MIT
"""

import json
import re
import glob

import pandas as pd
import requests
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime, timedelta
import pytz
from pandas import DataFrame
from tqdm import tqdm
import os
import shutil
from dataclasses import dataclass
import tempfile


@dataclass
class CSTag:
    """Class to represent a Computer Science category tag on arXiv."""
    code: str
    description: str


# Defined CS categories
CS_CATEGORIES = {
    "AI": CSTag("cs.AI", "Artificial Intelligence"),
    "MA": CSTag("cs.MA", "Multiagent Systems"),
    "CV": CSTag("cs.CV","Computer vision and pattern recognition"),
    # "LG": CSTag("cs.LG", "Machine Learning"),
    "CL": CSTag("cs.CL", "Computation and Language"),
    "IR": CSTag("cs.IR", "Information Retrieval"),
    # "NE": CSTag("cs.NE", "Neural and Evolutionary Computing"),
    "SE": CSTag("cs.SE", "Software Engineering")
}


def handle_category(df: DataFrame):
    """
    Categorize research papers based on keyword matching in titles for AI/ML research.

    Designed for AI engineers and data scientists to easily filter relevant papers.

    Args:
        df (pandas.DataFrame): DataFrame containing paper information

    Returns:
        pandas.DataFrame: DataFrame with an additional 'category' column
    """
    result_df = df.copy()
    result_df['category'] = 'other'

    # Define categories with their respective keywords
    categories = {
        # Review/overview papers
        'survey': ['survey'],
        'review': ['review', 'overview'],
        'tutorial': ['tutorial', 'introduction', 'primer', 'guide'],

        # Evaluation papers
        'bench': ['benchmark', 'leaderboard',"benchmarks"],
        'evaluation': ['evaluation', 'assessment', 'measuring', 'performance'],
        'comparison': ['comparison', 'comparative', 'versus', 'vs'],

        # Resources
        'dataset': ['dataset', 'corpus', 'data', 'database'],
        'tool': ['tool', 'toolkit', 'library', 'software', 'platform', 'package'],

        # Methodological papers
        'method': ['method', 'approach'],
        'framework': ['framework'],
        'model': ['model', 'modeling', 'neural'],
        'algorithm': ['algorithm', 'algorithmic'],
        'architecture': ['architecture'],
        'implementation': ['implementation', 'implementing'],
        'technique': ['technique'],
        'pipeline': ['pipeline'],

        # Improvement papers
        'optimization': ['optimization', 'optimizing', 'efficient', 'efficiency'],
        'improvement': ['improvement', 'improving', 'enhanced', 'enhancing', 'better'],
        'extension': ['extension', 'extending', 'augmentation'],
        'scaling': ['scaling', 'scale', 'large-scale', 'scalable'],

        # Problem-solving papers
        'solution': ['solution', 'solving'],
        'challenge': ['challenge', 'challenging', 'difficult', 'problem'],

        # Applied papers
        'application': ['application', 'applying', 'applied'],
        'case study': ['case study', 'case-study', 'real-world'],
        'deployment': ['deployment', 'production', 'industry'],

        # Analysis papers
        'analysis': ['analysis', 'analyzing', 'analytical'],
        'study': ['study', 'investigation', 'investigating'],
        'experiment': ['experiment', 'experimental', 'empirical'],
        'exploration': ['exploration', 'exploring', 'exploratory'],

        # Specialty areas
        'interpretability': ['interpretability', 'explainable', 'xai', 'explanation'],
        'uncertainty': ['uncertainty', 'confidence', 'probabilistic', 'bayesian'],
        'fairness': ['fairness', 'bias', 'ethical', 'ethics'],
        'robustness': ['robustness', 'robust', 'adversarial', 'defense'],
        'transfer': ['transfer', 'adaptation', 'domain adaptation', 'fine-tuning'],
        'multimodal': ['multimodal', 'multi-modal', 'cross-modal'],
        'generative': ['generative', 'generation', 'synthesis', 'synthetic'],
        'distributed': ['distributed', 'federated', 'decentralized'],
        'few-shot': ['few-shot', 'zero-shot', 'one-shot', 'meta-learning'],
        'representation': ['representation', 'embedding', 'feature'],
        'attention': ['attention', 'transformer', 'self-attention'],
        'causality': ['causality', 'causal', 'cause'],
        'reinforcement': ['reinforcement', 'rl', 'policy', 'reward'],
        'graph': ['graph', 'gnn', 'network'],
        'privacy': ['privacy', 'anonymity', 'secure', 'security'],
        'compression': ['compression', 'compressed', 'quantization', 'distillation'],
        'continual': ['continual', 'lifelong', 'incremental', 'online'],
        'foundation': ['foundation', 'foundational', 'base', 'baseline'],
        'pretraining': ['pretraining', 'pretrained', 'pre-trained', 'self-supervised']
    }

    # Apply categorization
    for category, keywords in categories.items():
        for keyword in keywords:
            mask = result_df['title'].str.contains(r'\b' + keyword + r'\b', case=False, regex=True)
            result_df.loc[mask & (result_df['category'] == 'other'), 'category'] = category

    return result_df


def find_github_url(text):
    """
    Find GitHub URL in the given text.

    Args:
        text (str): Text to search for GitHub URL

    Returns:
        str: GitHub URL if found, else empty string
    """
    if not text or not isinstance(text, str):
        return ""

    github_pattern = r'https?://(?:www\.)?github\.com/[^\s\)\"\']+|github\.com/[^\s\)\"\']+'
    match = re.search(github_pattern, text)
    url = match.group(0) if match else ""
    return url[:-1] if url.endswith(".") else url

def fetch_arxiv_papers_batch(start_date, end_date, categories=None, max_results_per_category=5000, download_pdfs=False, pdf_dir=None, user_timezone="Asia/Kolkata"):
    """
    Fetch arXiv papers for a specific date range.

    Parameters
    ----------
    start_date : datetime
        Start date for the search
    end_date : datetime
        End date for the search
    categories : list of str or CSTag, optional
        Category codes (e.g., "cs.AI") or `CSTag` objects.
        If None, all predefined CS categories are used.
    max_results_per_category : int, default=5000
        Maximum number of results to fetch per category.
    download_pdfs : bool, default=False
        Whether to download PDF files for each paper.
    pdf_dir : str, optional
        Directory to save downloaded PDFs. Defaults to "./arxiv_pdfs".
        Only used if `download_pdfs=True`.
    user_timezone : str, default="Asia/Kolkata"
        Timezone for aligning daily arXiv updates (IANA format).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with paper information for the date range
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("arxiv_fetcher")

    # Setup PDF directory if downloading is enabled
    if download_pdfs:
        if pdf_dir is None:
            pdf_dir = "./arxiv_pdfs"
        os.makedirs(pdf_dir, exist_ok=True)
        logger.info(f"PDFs will be downloaded to: {pdf_dir}")

    # If no categories specified, use all defined categories
    if categories is None:
        categories = list(CS_CATEGORIES.values())

    # Normalize categories to CSTag objects
    category_tags = []
    for cat in categories:
        if isinstance(cat, str):
            # Check if it's a key in CS_CATEGORIES
            if cat in CS_CATEGORIES:
                category_tags.append(CS_CATEGORIES[cat])
            else:
                # Assume it's a category code like "cs.AI"
                for tag in CS_CATEGORIES.values():
                    if tag.code == cat:
                        category_tags.append(tag)
                        break
                else:
                    # Not found, create a new CSTag
                    category_tags.append(CSTag(cat, cat))
        else:
            # Assume it's already a CSTag
            category_tags.append(cat)

    # Generate dates for the range
    search_dates = []
    current_date = end_date
    while current_date >= start_date:
        search_dates.append(current_date.strftime("%Y%m%d"))
        current_date -= timedelta(days=1)

    logger.info(f"Searching for papers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}: {', '.join(search_dates)}")

    # Dictionary to hold results for each category
    results = {}

    # Process each category
    for category in category_tags:
        logger.info(f"Fetching papers for {category.code} ({category.description})")

        # Build date search part of the query
        date_query = " OR ".join([f"submittedDate:[{date}0000 TO {date}2359]" for date in search_dates])
        query = f"cat:{category.code} AND ({date_query})"

        # Fetch results
        abstracts = []
        total_fetched = 0
        batch_size = 100  # arXiv API limit per request

        with tqdm(total=min(max_results_per_category, batch_size),
                  desc=f"Fetching {category.code} papers") as pbar:
            while total_fetched < max_results_per_category:
                current_batch = min(batch_size, max_results_per_category - total_fetched)

                # Construct query URL
                base_url = "http://export.arxiv.org/api/query"
                params = {
                    "search_query": query,
                    "start": total_fetched,
                    "max_results": current_batch,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending"
                }

                try:
                    # Make request
                    response = requests.get(base_url, params=params)
                    response.raise_for_status()

                    # Parse XML response
                    root = ET.fromstring(response.content)

                    # Get total results count
                    namespace = {'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
                                 'atom': 'http://www.w3.org/2005/Atom',
                                 'arxiv': 'http://arxiv.org/schemas/atom'}

                    # Update total if this is the first batch
                    if total_fetched == 0:
                        total_results_elem = root.find('.//opensearch:totalResults', namespace)
                        if total_results_elem is not None and total_results_elem.text:
                            total_results = int(total_results_elem.text)
                            # Update progress bar total
                            pbar.total = min(max_results_per_category, total_results)
                            pbar.refresh()

                    # Find entries and extract data
                    entries = root.findall('.//atom:entry', namespace)

                    # If no entries, break the loop
                    if not entries:
                        break

                    # Process each entry
                    for entry in entries:
                        try:
                            # Get ID
                            id_url = entry.find('./atom:id', namespace).text
                            arxiv_id = id_url.split('/abs/')[1] if '/abs/' in id_url else id_url

                            # Get title and remove extra whitespace
                            title_elem = entry.find('./atom:title', namespace)
                            title = " ".join(
                                title_elem.text.split()) if title_elem is not None and title_elem.text else "No title"

                            # Get abstract and remove extra whitespace
                            summary_elem = entry.find('./atom:summary', namespace)
                            abstract = " ".join(
                                summary_elem.text.split()) if summary_elem is not None and summary_elem.text else "No abstract"

                            # Get published date
                            published_elem = entry.find('./atom:published', namespace)
                            published = published_elem.text if published_elem is not None else None

                            # Get updated date
                            updated_elem = entry.find('./atom:updated', namespace)
                            updated = updated_elem.text if updated_elem is not None else None

                            # Get authors
                            authors = []
                            for author_elem in entry.findall('./atom:author/atom:name', namespace):
                                if author_elem.text:
                                    authors.append(author_elem.text)

                            # Get categories/tags
                            categories_list = []
                            for cat_elem in entry.findall('./atom:category', namespace):
                                if cat_elem.get('term'):
                                    categories_list.append(cat_elem.get('term'))

                            # Get PDF link
                            pdf_link = None
                            for link in entry.findall('./atom:link', namespace):
                                if link.get('title') == 'pdf':
                                    pdf_link = link.get('href')

                            # Create paper info dictionary
                            paper_info = {
                                'arxiv_id': arxiv_id,
                                'title': title,
                                'abstract': abstract,
                                'authors': authors,
                                'categories': categories_list,
                                'published': published,
                                'updated': updated,
                                'pdf_url': pdf_link,
                                'url': f"https://arxiv.org/abs/{arxiv_id}"
                            }

                            # Download PDF if requested
                            if download_pdfs and pdf_link:
                                pdf_filename = f"{arxiv_id.replace('/', '_')}.pdf"
                                pdf_path = os.path.join(pdf_dir, pdf_filename)

                                if not os.path.exists(pdf_path):
                                    try:
                                        # Download PDF with a separate request
                                        pdf_response = requests.get(pdf_link, stream=True)
                                        pdf_response.raise_for_status()

                                        with open(pdf_path, 'wb') as pdf_file:
                                            shutil.copyfileobj(pdf_response.raw, pdf_file)

                                        # Add local PDF path to paper info
                                        paper_info['local_pdf_path'] = pdf_path
                                        logger.debug(f"Downloaded PDF: {pdf_filename}")

                                        # Add small delay between PDF downloads to be nice to the server
                                        time.sleep(0.5)

                                    except Exception as e:
                                        logger.error(f"Error downloading PDF for {arxiv_id}: {str(e)}")
                                else:
                                    # PDF already exists
                                    paper_info['local_pdf_path'] = pdf_path
                                    logger.debug(f"PDF already exists: {pdf_filename}")

                            abstracts.append(paper_info)

                        except Exception as e:
                            logger.error(f"Error parsing entry: {str(e)}")

                    # Update total fetched and progress bar
                    batch_fetched = len(entries)
                    total_fetched += batch_fetched
                    pbar.update(batch_fetched)

                    # If we got fewer results than requested, we're done
                    if batch_fetched < current_batch:
                        break

                    # Apply rate limiting between batches
                    if total_fetched < max_results_per_category:
                        # Display rate limiting info in the progress bar description
                        pbar.set_description(f"Rate limiting (3s)")
                        time.sleep(3)
                        pbar.set_description(f"Fetching {category.code} papers")

                except Exception as e:
                    logger.error(f"Error fetching batch: {str(e)}")
                    time.sleep(3)  # Wait before retrying

        # Store results for this category
        results[category.code] = abstracts
        logger.info(f"Found {len(abstracts)} papers for {category.code}")

        # Rate limit between categories
        if category != category_tags[-1]:  # Not the last category
            time.sleep(3)

    total_papers = sum(len(papers) for papers in results.values())
    logger.info(f"Completed fetching a total of {total_papers} papers across {len(category_tags)} categories")

    data_frame_data = []
    for k, v in results.items():
        for vv in v:
            data_frame_data.append({**vv, "tag": k})

    df = pd.DataFrame(data_frame_data)
    df_with_categories = handle_category(df)
    df_with_categories['github_url'] = df_with_categories['abstract'].apply(find_github_url)
    df_cleaned = df_with_categories.drop_duplicates(subset=['title'], keep='first')
    return df_cleaned


def merge_batch_csv_files(output_file="final.csv", batch_pattern="arxiv_papers_batch_*.csv"):
    """
    Merges all batch CSV files into a single final CSV file.

    Parameters
    ----------
    output_file : str, default="final.csv"
        Name of the final merged CSV file
    batch_pattern : str, default="arxiv_papers_batch_*.csv"
        Pattern to match batch CSV files

    Returns
    -------
    str : Path to the merged CSV file
    """

    # Find all batch CSV files
    batch_files = glob.glob(batch_pattern)

    if not batch_files:
        print(f"No batch files found matching pattern: {batch_pattern}")
        return None

    print(f"Found {len(batch_files)} batch files to merge: {batch_files}")

    # Read and concatenate all batch files
    dataframes = []
    for batch_file in batch_files:
        df = pd.read_csv(batch_file)
        dataframes.append(df)
        print(f"Loaded {len(df)} rows from {batch_file}")

    # Concatenate all dataframes
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        print(f"Total rows before deduplication: {len(merged_df)}")

        # Remove duplicates based on title
        merged_df = merged_df.drop_duplicates(subset=['title'], keep='first')
        print(f"Total rows after deduplication: {len(merged_df)}")

        # Save to final CSV
        merged_df.to_csv(output_file, index=False)
        print(f"Merged data saved to {output_file}")

        return output_file
    else:
        print("No dataframes to merge")
        return None


def fetch_latest_arxiv_cs_papers(
    categories: List[Union[str, CSTag]] = None,
    max_results_per_category: int = 5000,
    download_pdfs: bool = False,
    pdf_dir: Optional[str] = None,
    user_timezone: str = "Asia/Kolkata",
    days: int = 90,
    batch_size_days: int = 30
) -> List[Dict]:
    """
    Fetch the latest arXiv papers from selected Computer Science categories.

    This function queries the arXiv API, retrieves metadata for recent papers,
    assigns content-based categories (via keyword matching), extracts GitHub URLs,
    and optionally downloads PDFs. Duplicate titles are removed.

    For large date ranges (>30 days), the function processes data in batches
    and saves each batch to a separate CSV file before merging.

    Parameters
    ----------
    categories : list of str or CSTag, optional
        Category codes (e.g., "cs.AI") or `CSTag` objects.
        If None, all predefined CS categories are used.
    max_results_per_category : int, default=1000
        Maximum number of results to fetch per category.
    download_pdfs : bool, default=False
        Whether to download PDF files for each paper.
    pdf_dir : str, optional
        Directory to save downloaded PDFs. Defaults to "./arxiv_pdfs".
        Only used if `download_pdfs=True`.
    user_timezone : str, default="Asia/Kolkata"
        Timezone for aligning daily arXiv updates (IANA format).
    days : int, default=30
        Number of past days to include in the search.
    batch_size_days : int, default=30
        Number of days to process in each batch to prevent memory overload.

    Returns
    -------
    papers : list of dict
        A list of paper metadata dictionaries, one per paper, with keys:
        - arxiv_id (str): Unique arXiv identifier.
        - title (str): Paper title.
        - abstract (str): Paper abstract.
        - authors (list[str]): Author names.
        - categories (list[str]): arXiv category codes.
        - published (str): Publication datetime (ISO format).
        - updated (str): Last updated datetime (ISO format).
        - pdf_url (str): Direct PDF download link.
        - url (str): Abstract page URL.
        - tag (str): Primary category tag (e.g., "cs.AI").
        - category (str): Auto-assigned content category (keyword-based).
        - github_url (str): Extracted GitHub URL from abstract (if found).
        - local_pdf_path (str, optional): Path to local PDF file if downloaded.

    Examples
    --------
    >>> papers = fetch_latest_arxiv_cs_papers(["cs.AI"], days=3)
    >>> print(f"Found {len(papers)} papers")
    >>> print(papers[0]["title"])
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("arxiv_fetcher")

    # Get current datetime in user's timezone
    user_tz = pytz.timezone(user_timezone)
    now_in_user_tz = datetime.now(user_tz)

    # arXiv typically announces new papers at 8 PM ET (5:30 AM IST next day)
    et_tz = pytz.timezone("US/Eastern")
    arxiv_update_time_et = datetime.now(et_tz).replace(hour=20, minute=0, second=0, microsecond=0)

    # Convert arXiv update time to user timezone
    arxiv_update_time_user_tz = arxiv_update_time_et.astimezone(user_tz)

    # Determine which dates to search for papers
    # Determine base offset depending on release time
    if now_in_user_tz.hour < 6:
        start_offset = 1  # before daily update → start from yesterday
    else:
        start_offset = 0  # after daily update → start from today

    # Calculate start and end dates for the full range
    end_date = now_in_user_tz - timedelta(days=start_offset)
    start_date = end_date - timedelta(days=days-1)  # -1 to make it inclusive

    logger.info(f"Fetching papers for {days} days from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # If days <= batch_size_days, process directly without batching
    if days <= batch_size_days:
        df = fetch_arxiv_papers_batch(
            start_date=start_date,
            end_date=end_date,
            categories=categories,
            max_results_per_category=max_results_per_category,
            download_pdfs=download_pdfs,
            pdf_dir=pdf_dir,
            user_timezone=user_timezone
        )
        df_cleaned = df.drop_duplicates(subset=['title'], keep='first')
        return df_cleaned
    else:
        batch_dfs = []
        current_start = start_date
        batch_num = 1

        while current_start <= end_date:
            # Calculate the end date for this batch
            current_end = min(current_start + timedelta(days=batch_size_days-1), end_date)

            logger.info(f"Processing batch {batch_num}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")

            # Fetch papers for this batch
            batch_df = fetch_arxiv_papers_batch(
                start_date=current_start,
                end_date=current_end,
                categories=categories,
                max_results_per_category=max_results_per_category,
                download_pdfs=download_pdfs,
                pdf_dir=pdf_dir,
                user_timezone=user_timezone
            )

            # Save this batch to a CSV file named with the date range
            batch_filename = f"arxiv_papers_batch_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}.csv"
            batch_df.to_csv(batch_filename, index=False)
            logger.info(f"Saved batch to {batch_filename}")

            batch_dfs.append(batch_df)

            # Move to the next batch
            current_start = current_end + timedelta(days=1)
            batch_num += 1

        # Combine all batches
        if batch_dfs:
            combined_df = pd.concat(batch_dfs, ignore_index=True)
            df_cleaned = combined_df.drop_duplicates(subset=['title'], keep='first')
            logger.info(f"Combined {len(batch_dfs)} batches with a total of {len(df_cleaned)} unique papers")
            return df_cleaned
        else:
            # Return empty dataframe if no batches were processed
            return pd.DataFrame()

import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fetch arXiv CS papers")
    parser.add_argument("--categories", default=None,
                        help="Comma-separated category codes, e.g. AI,MA,CV (default: all)")
    parser.add_argument("--past", type=int, default=5,
                        help="Number of past days to fetch (default: 5)")
    parser.add_argument("--max", type=int, default=5000,
                        help="Max results per category (default: 5000)")
    parser.add_argument("--batch", type=int, default=30,
                        help="Batch size in days (default: 30)")
    parser.add_argument("-o", "--output", default="final.csv",
                        help="Output CSV file path (default: final.csv)")
    args = parser.parse_args()

    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None

    logging.disable(logging.CRITICAL)

    df = fetch_latest_arxiv_cs_papers(
        categories=cats,
        days=args.past,
        batch_size_days=args.batch,
        max_results_per_category=args.max,
    )

    output_path = os.path.abspath(args.output)
    df.to_csv(output_path, index=False)
    print(output_path)
