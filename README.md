# SummerShelf
Summer Shelf book recommender app created for the Maven Bookshelf Challenge

SummerShelf is an interactive book recommender app designed to help readers build their ideal summer reading list. Users can set preferences, view personalised recommendations, and explore review summaries to find their perfect book.

## Install Requirements
pip install -r requirements.txt

## Run the Streamlit App
streamlit run summer_shelf.py

## Deployment
This app is deployed publicly on Streamlit. The code is available here in GitHub. 

## Data
The app uses the following pre-processed datasets, available in the data folder:
- books_dataset.csv: a unique list of books and key attributes
- book_rating_distribution.csv: unpivoted review data for rating breakdown visualisation
- review_summary_per_book.csv: aggregated summary data from itemised reviews
- review_subset.csv: selection of up to 15 itemised reviews per book

