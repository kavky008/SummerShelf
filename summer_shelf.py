"""
summer_shelf.py

A streamlit app that recommends books based on user preferences, and allows
users to build and export their reading list to csv. 

To run:
    Run this command with streamlit: streamlit run summer_shelf.py

Author: Kerryn Gresswell

"""

# Import required libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import ast

# -------------------------------------------------------------------------------
# --------------------------- PAGE CONFIGURATION --------------------------------
# -------------------------------------------------------------------------------

st.set_page_config(page_title="Summer Shelf", layout="wide")
pink = "#f92f47"
light_pink = "rgba(249, 47, 71, 0.07)"

# -------------------------------------------------------------------------------
# ------------------------------ LOAD DATA --------------------------------------
# -------------------------------------------------------------------------------

@st.cache_data
def load_books():
    df = pd.read_csv("data/books_dataset.csv")

    # Function to convert string to actual Python lists
    def try_parse_list(x):
        try:
            return ast.literal_eval(x)
        # if it fails, return empty list rather than error
        except Exception:
            return [] 
        
    df['genres_list'] = df['genres_list'].apply(try_parse_list)
    df['theme_label'] = df.get('theme_label', '').fillna("")

    return df

@st.cache_data
def load_ratings():
    return pd.read_csv("data/book_rating_distribution.csv")

@st.cache_data
def load_review_summaries():
    return pd.read_csv("data/review_summary_per_book.csv")

@st.cache_data
def load_reviews():
    return pd.read_csv("data/review_subset.csv")

books = load_books()
ratings = load_ratings()
review_summaries = load_review_summaries()
reviews = load_reviews()

# -------------------------------------------------------------------------------
# -------------------- SESSION STATE INITIALISATION -----------------------------
# -------------------------------------------------------------------------------

# Set default show books value to 10
if 'show_count' not in st.session_state:
    st.session_state.show_count = 10

# Create an empty dataframe to hold the filtered books
if 'filtered_books' not in st.session_state:
    st.session_state.filtered_books = pd.DataFrame()

# Track when to show the results page vs default search page
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# Track whether the user selects a mystery trip
if "random_trip" not in st.session_state:
    st.session_state.random_trip = False

# Create an empty list to store books added to suitcase
if 'suitcase' not in st.session_state:
    st.session_state.suitcase = []

# Track which books have had see more like this expanded
if 'similar_shown' not in st.session_state:
    st.session_state.similar_shown = set()


# -------------------------------------------------------------------------------
# ----------------------------- RATING VISUAL -----------------------------------
# -------------------------------------------------------------------------------

def create_rating_sparkline(df):
    """
    Creates a clean column chart (sparkline style with no labels etc.) of the rating
    breakdown by rating score (1-5). Generates a matplotlib chart and returns as an
    image.
    """

    fig, ax = plt.subplots(figsize=(2.5, 1.0), dpi=120)

    # Make sure ratings are shown in correct order
    df = df.sort_values('rating')

    # Plot a bar for each rating category and the percentage of reviews
    ax.bar(df['rating'].astype(str), df['percent'], color='#a783c6', edgecolor='none', width=0.8)

    # Show only bottom axis and hide other elements for a clean visual look 
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#dddddd')
    ax.spines['bottom'].set_linewidth(0.5)
    for side in ['top', 'left', 'right']:
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    
    # Save the plot as png 
    plt.tight_layout(pad=0.1)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.08, transparent=True)
    plt.close(fig)
    buf.seek(0)

    return buf


# -------------------------------------------------------------------------------
# --------------------- DISPLAY BOOK DETAILS FUNCTION----------------------------
# -------------------------------------------------------------------------------

def display_book(row, unique_key, ratings, review_summaries, reviews, books, highlighted=False):
    """
    Displays a single book in the app, including:
     - Cover image and summary stats
     - Description
     - Review section with rating stats and itemised reviews
     - Buttons to add to suitcase or remove from list
     - Button to see similar books and section to display if selected

     Does not return anything - output is displayed to the app. 
    """

    with st.expander(f"{row['original_title']} by {row['author']} ({row['avg_rating']:.1f} ⭐)"):
        left, right = st.columns([1.1, 3])

        with left:
            # Display book cover
            if pd.notna(row['image_url']) and row['image_url'].strip():
                st.image(row['image_url'], width=120)

        with right:
            # Display book summary with book length, genres/themes, publication era
            st.markdown(
                f"**Pages:** {row['num_pages']}  |  ⏱️ ≈ {row.get('est_reading_hours', 'N/A')} hrs  |  🧳 {row.get('length_category', 'N/A')}"
            )

            genres = row.get('genres_list', [])
            if isinstance(genres, list) and genres:
                genres_str = ", ".join(genres)
            else:
                genres_str = "N/A"

            st.markdown(f"**Theme:** {row.get('theme_label', 'N/A')}  \n**Genres:** {genres_str}")

            st.markdown(
                f"**Published:** {row.get('original_publication_year', 'N/A')} ({row.get('publication_era', 'N/A')})"
            )

        # Divider line
        st.markdown("---")

        # Expander section with given book's description
        desc = row.get('description', '')
        with st.expander("What's it about?", expanded=False):
            if pd.notna(desc) and desc.strip():
                st.write(desc)
            else:
                st.write("No description available.")

        # Expander section for review summary
        with st.expander("See what readers are saying", expanded=False):

            # Fetch ratings and review summary rows for given work id
            book_ratings = ratings[ratings['work_id'] == row['work_id']]
            summary_row = review_summaries[review_summaries['work_id'] == row['work_id']]

            if not book_ratings.empty:
                left_col2, right_col2 = st.columns([2.5, 1.5])

                with left_col2:
                    # Display rating and review snapshot
                    st.markdown("**Ratings & Reviews**")
                    st.markdown(f"⭐ {row['rating_category']} ({row['avg_rating']:.1f}) from {row['ratings_count']:,} ratings")
                    st.markdown(f"✏️ {row.get('text_reviews_count', 0):,} readers wrote a review")
                    if not summary_row.empty and pd.notna(summary_row.iloc[0].get('dnf_percent')):
                        dnf_pct = summary_row.iloc[0]['dnf_percent']
                        st.markdown(f"🚫 {dnf_pct:.1f}% of reviewers said they did not finish this book")

                with right_col2:
                    # Display rating breakdown visual
                    st.markdown("**Rating Breakdown**")
                    sparkline = create_rating_sparkline(book_ratings)
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #e8d9ef; border-radius: 8px; padding: 8px; width: 200px; margin-top: 6px;">
                            <img src="data:image/png;base64,{base64.b64encode(sparkline.read()).decode()}" 
                                style="display: block; margin: 0 auto; width: 100%;" />
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Display featured reviews
                if not summary_row.empty:
                    review = summary_row.iloc[0]

                    if pd.notna(review.get('featured_positive_review')) and review['featured_positive_review'].strip():
                        st.markdown(
                            f"""
                            <div style="margin-top: 12px; padding: 10px; background-color: #f2fbf6; border-radius: 5px;">
                                <strong>This reader loved it:</strong><br>{review['featured_positive_review']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if pd.notna(review.get('featured_negative_review')) and review['featured_negative_review'].strip():
                        st.markdown(
                            f"""
                            <div style="margin-top: 12px; padding: 10px; background-color: #faeef0; border-radius: 5px;">
                                <strong>But it wasn't for everyone...</strong><br>{review['featured_negative_review']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if pd.notna(review.get('most_liked_review')) and review['most_liked_review'].strip():
                        rating = review.get('most_liked_rating', 0)
                        stars = "⭐" * int(rating if not pd.isna(rating) else 0)
                        likes = (
                            f"({int(review.get('most_liked_likes', 0))} likes)"
                            if pd.notna(review.get('most_liked_likes'))
                            else ""
                        )

                        st.markdown(
                            f"""
                            <div style="margin-top: 12px; padding: 10px; background-color: #f3f3f3; border-radius: 5px;">
                                <strong>Most popular opinion:</strong><br>
                                {stars} {likes}<br>
                                {review['most_liked_review']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Extra space between review and bottom of expander box
                    st.markdown("")

                    # Expander section to see itemised reviews for given book
                    with st.expander("Still not sure? See more reviews."):
                        display_reviews(row['work_id'], reviews)
            else:
                st.write("No reviews available.")

        # BOOK ACTION BUTTONS
        btn1, btn2, btn3 = st.columns(3)

        with btn1:
            # Append key book details to suitcase list when button clicked
            if st.button("Add to Suitcase", key=f"add_{unique_key}", use_container_width=True):
                if row['work_id'] not in [b['work_id'] for b in st.session_state.suitcase]:
                    st.session_state.suitcase.append({
                        'work_id': row['work_id'],
                        'Title': row['original_title'],
                        'Author': row['author'],
                        'est_reading_hours': row.get('est_reading_hours', None)
                    })
                    st.success(f"Added '{row['original_title']}' to your suitcase!")
                    st.rerun()

        with btn2:
            # Remove given book from filtered books if not for me clicked
            if st.button("Not For Me", key=f"not_{unique_key}", use_container_width=True):
                st.session_state.filtered_books = st.session_state.filtered_books[
                    st.session_state.filtered_books['work_id'] != row['work_id']
                ].reset_index(drop=True)
                st.rerun()

        with btn3:
            # Add book to similar shown set and rerun
            if st.button("See More Like This", key=f"more_{unique_key}", use_container_width=True):
                st.session_state.similar_shown.add(unique_key)
                st.rerun()

        # Check if see more like this was checked for this book
        if unique_key in st.session_state.similar_shown:

            similar_ids = []

            # Try to parse the similar books list as Python list or return empty list if error
            if pd.notna(row.get('similar_books_list')) and row['similar_books_list'].strip() not in ("[]", ""):
                try:
                    similar_ids = ast.literal_eval(row['similar_books_list'])
                except Exception:
                    similar_ids = []

            # If similar books exist, display the similar books expander and book details
            if similar_ids:
                with st.expander("Similar Books", expanded=unique_key in st.session_state.similar_shown):
                    for sid in similar_ids:
                        match = books[books['work_id'] == sid]
                        if not match.empty:
                            match_row = match.iloc[0]
                            st.markdown(
                                f"**{match_row['original_title']}** by *{match_row['author']}* ({match_row['avg_rating']:.1f} ⭐)"
                            )
                            if st.button(
                                "Add to Suitcase", key=f"add_similar_{sid}_{unique_key}", use_container_width=True
                            ):
                                if match_row['work_id'] not in [b['work_id'] for b in st.session_state.suitcase]:
                                    st.session_state.suitcase.append({
                                        'work_id': match_row['work_id'],
                                        'Title': match_row['original_title'],
                                        'Author': match_row['author'],
                                        'est_reading_hours': match_row.get('est_reading_hours', None)
                                    })
                                    st.success(f"Added '{match_row['original_title']}' to your suitcase!")
            else:
                st.info("No similar books found.")


# -------------------------------------------------------------------------------
# ------------------------ DISPLAY REVIEWS FUNCTION------------------------------
# -------------------------------------------------------------------------------

def display_reviews(work_id, reviews, reviews_per_page=5):
    """
    Displays itemised reviews for the given book to the streamlit app. 
    Shows a limited number of books but users can show more to move
    to next page. Doesn't return anything as output is displayed
    directly to the app.
    """

    # Filter reviews to the given work id and not empty
    filtered = reviews[reviews['work_id'] == work_id].dropna(subset=['review_text'])

    # Sort by rating descending order
    filtered = filtered.sort_values(by="rating", ascending=False)
    
    # Calculate how many reviews are available for this book
    total_reviews = len(filtered)
    
    if total_reviews == 0:
        st.write("No additional reviews found.")
        return
    
    # Track which page of review results user is on and which reviews to show
    page_key = f"review_page_{work_id}"
    page_num = st.session_state.get(page_key, 0)
    start_idx = page_num * reviews_per_page
    end_idx = start_idx + reviews_per_page
    to_display = filtered.iloc[start_idx:end_idx]

    # Display the review details
    for _, row in to_display.iterrows():
        stars = "⭐" * int(row['rating']) if not pd.isna(row['rating']) else ""
        st.markdown(f"""
            <div style="margin-top: 12px; padding: 10px; background-color: #f3f3f3; border-radius: 5px;">
                {stars}<br>
                {row['review_text']}
            </div>
        """, unsafe_allow_html=True)
    st.markdown("")

    # If more reviews still exist, display the show more button
    if end_idx < total_reviews:
        if st.button("Show more reviews", key=f"more_{work_id}_{page_num}"):
            # Increment page number by one when clicked
            st.session_state[page_key] = page_num + 1


# -------------------------------------------------------------------------------
# ------------------------- BANNER IMAGE FOR PAGE -------------------------------
# -------------------------------------------------------------------------------

st.image("summer_shelf_banner.jpg", use_container_width=True)


# -------------------------------------------------------------------------------
# --------------------------FILTER OPTIONS --------------------------------------
# -------------------------------------------------------------------------------

# Prepare sorted list of unique options for filter dropdowns
all_genres = sorted({genre for sublist in books['genres_list'] for genre in sublist})
all_themes = sorted(books['theme_label'].dropna().unique())
all_pub_periods = sorted(books['publication_era'].dropna().unique())


# --------------------------------------------------------------------------------
# --------------------- INITIAL TRIP PLANNER PAGE VIEW ---------------------------
# --------------------------------------------------------------------------------

if not st.session_state.show_results:

    # Split into three equal width columns
    col1, col2, col3 = st.columns(3, gap="medium")

    # --------------------- COL 1 (LEFT): INTRO ----------------------------------
    with col1:
        st.markdown(f"""
            <div style="
                background-color: {light_pink};
                border: 2px solid {pink};
                border-radius: 10px;
                padding: 1.2rem;
                margin-top: 1rem;
                font-size: 1rem;
                line-height: 1.5;
            ">
                <h4 style="margin-top: 0; margin-bottom: 0;">Welcome to Summer Shelf!</h4>
                Whether you're after a quick beach read or a book to sink into all holiday long, we'll help you find your next great escape.<br><br>
                <b>How it works</b><br>
                First, design your adventure - choose what kind of reads you're in the mood for, and refine your trip with extra options or searches if you want to narrow it down. 
                <br><br>
                When you're ready, hit <b>Start Your Adventure</b> to get your personalised recommendations. <br><br>
                Feeling spontaneous? Leave everything blank, hit <b>Take a Mystery Trip</b> and let us surprise you.
            </div>
        """, unsafe_allow_html=True)

    # --------------------- COL 2 (MIDDLE): MAIN FILTERS -------------------------
    with col2:
        st.markdown("<div style='padding-top: 2.3rem'></div>", unsafe_allow_html=True)
        st.markdown("#### Choose your adventure")

        length_options = [
            "Day Trip (under 250 pages, ~1–4 hrs)",
            "Long Weekend (250–400 pages, ~4–8 hrs)",
            "Epic Journey (400+ pages, 8+ hrs)"
        ]

        selected_lengths = st.multiselect("Length", options=length_options)
        selected_genres = st.multiselect("Genres", all_genres)
        selected_themes = st.multiselect("Themes", all_themes)
        selected_pub_period = st.multiselect("Publication Era", all_pub_periods)

    # ---------------------- COL 3 (RIGHT): REFINE & SEARCH -----------------------
    with col3:
        st.markdown("<div style='padding-top: 2.3rem'></div>", unsafe_allow_html=True)
        st.markdown("#### Refine your trip")

        min_rating = st.number_input("Minimum Avg Rating", min_value=0.0, max_value=5.0, value=4.0, step=0.1, format="%.1f")
        min_reviews = st.number_input("Minimum Number of Reviews", min_value=0, value=100)
        
        search_title = st.text_input("Search by Book Title")
        search_author = st.text_input("Search by Author")

        # Display buttons side by side
        with st.container():
            col3a, col3b = st.columns([1,1])

            with col3a:
                if st.button("Start Your Adventure", use_container_width=True):
                    # Store all filter variables to session state
                    st.session_state.selected_lengths = selected_lengths
                    st.session_state.selected_genres = selected_genres
                    st.session_state.selected_themes = selected_themes
                    st.session_state.selected_pub_period = selected_pub_period
                    st.session_state.min_rating = min_rating
                    st.session_state.min_reviews = min_reviews
                    st.session_state.search_title = search_title
                    st.session_state.search_author = search_author
                    st.session_state.random_trip = False

                    # Save copy of the filtered data to session state
                    filtered = books.copy()

                    # Map the length category filter to page lengths & filter dataset
                    length_conditions = []

                    for length in selected_lengths:
                        if "Day Trip" in length:
                            length_conditions.append((filtered['num_pages'] <= 250))
                        elif "Long Weekend" in length:
                            length_conditions.append((filtered['num_pages'] > 250) & (filtered['num_pages'] <= 400))
                        elif "Epic Journey" in length:
                            length_conditions.append((filtered['num_pages'] > 400))
                    if length_conditions:
                        length_filter = length_conditions[0]
                        for cond in length_conditions[1:]:
                            length_filter |= cond
                        filtered = filtered[length_filter]

                    # Apply other filters to dataset
                    if selected_genres:
                        filtered = filtered[filtered['genres_list'].apply(lambda gs: any(g in gs for g in selected_genres))]
                    if selected_themes:
                        filtered = filtered[filtered['theme_label'].isin(selected_themes)]
                    if selected_pub_period:
                        filtered = filtered[filtered['publication_era'].isin(selected_pub_period)]
                    if search_title:
                        filtered = filtered[filtered['original_title'].str.contains(search_title, case=False, na=False)]
                    if search_author:
                        filtered = filtered[filtered['author'].str.contains(search_author, case=False, na=False)]
                    filtered = filtered[filtered['avg_rating'] >= min_rating]
                    filtered = filtered[filtered['reviews_count'] >= min_reviews]

                    # Sort and save results, change show results to true to change page
                    filtered = filtered.sort_values(by="avg_rating", ascending=False).reset_index(drop=True)
                    st.session_state.filtered_books = filtered.copy(deep=True)
                    st.session_state.show_count = 10
                    st.session_state.show_results = True
                    st.rerun()

            with col3b:
                if st.button("Take a Mystery Trip", use_container_width=True):
                    # Store all filter variables to session state but mark as random trip
                    st.session_state.selected_lengths = selected_lengths
                    st.session_state.selected_genres = selected_genres
                    st.session_state.selected_themes = selected_themes
                    st.session_state.selected_pub_period = selected_pub_period
                    st.session_state.min_rating = min_rating
                    st.session_state.min_reviews = min_reviews
                    st.session_state.search_title = search_title
                    st.session_state.search_author = search_author
                    st.session_state.random_trip = True

                    # Randomly select 50 books, or if less than that in dataset copy full dataset
                    mystery_sample_size = 50
                    if len(books) > mystery_sample_size:
                        filtered = books.sample(n=mystery_sample_size, random_state=None).copy()
                    else:
                        filtered = books.copy()

                    # Sort and save the results and change show results to true to change page
                    filtered = filtered.sort_values(by="avg_rating", ascending=False).reset_index(drop=True)
                    st.session_state.filtered_books = filtered.copy(deep=True)
                    st.session_state.show_count = 10
                    st.session_state.show_results = True
                    st.rerun()

# -------------------------------------------------------------------------------
# ------------------------ RESULTS PAGE VIEW ------------------------------------
# -------------------------------------------------------------------------------

else:
    # Split into three columns with custom widths
    left_col, mid_col, right_col = st.columns([1.3, 2.9, 1.4], gap="medium")

    # Load in the saved filtered books data 
    filtered = st.session_state.filtered_books

    # ------------------------ LEFT COLUMN: INFO ---------------------------------
    with left_col:
        st.markdown(f"""
            <div style="
                background-color: {light_pink};
                border: 2px solid {pink};
                border-radius: 10px;
                padding: 1.2rem;
                margin-top: 1rem;
                font-size: 1rem;
                line-height: 1.5;
            ">
                Your recommendations are ready! <br><br>
                Expand the book titles to learn more about each book. <br><br>
                Like what you see? <b>Add it to your suitcase </b> or click <b> See more like this </b> for similar books.<br><br>
                Not quite right or already read it? Click <b>Not for me</b> to remove a book.<br>
                <br>
                Need to make a change? <b>Change plan</b> to head back and adjust your preferences, or hit <b>Restart trip</b> to start fresh.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1rem'></div>", unsafe_allow_html=True)

        # Display buttons side by side
        btn_col1, btn_col2 = st.columns(2, gap="small")

        with btn_col1:
            # Revert back to search page and clear any see more like this expander sections
            if st.button("Change plan", use_container_width=True):
                st.session_state.show_results = False
                st.session_state.similar_shown = set()

        with btn_col2:
            # Clear all session states and rerun to start from scratch on search page
            if st.button("Restart trip", use_container_width=True):
                st.session_state.clear()
                st.session_state.similar_shown = set()
                st.rerun()

    # ------------------- MIDDLE COLUMN: RECOMMENDATIONS ------------------
    with mid_col:
        st.markdown("### Your Recommendations")

        # Highlighted books section
        highlighted_ids = []
        highlighted_books = []

        # If there's enough books to display featured, select which books to show
        if len(filtered) >= 5:
            # Top rated (highest rating)
            top_rated = filtered.sort_values(by="avg_rating", ascending=False).iloc[0]
            highlighted_books.append(("Top Rated Stop", top_rated))
            highlighted_ids.append(top_rated["work_id"])

            # Most popular (most reviews)
            most_reviews = filtered.sort_values(by="reviews_count", ascending=False).iloc[0]
            if most_reviews["work_id"] not in highlighted_ids:
                highlighted_books.append(("Most Popular Destination", most_reviews))
                highlighted_ids.append(most_reviews["work_id"])

            # Hidden gem (highest rated 7-8k reviews preferred, or <10k if not found)
            gem_pool = filtered[
                (filtered["reviews_count"] >= 7000) & (filtered["reviews_count"] <= 8000)
            ]
            if gem_pool.empty:
                gem_pool = filtered[filtered["reviews_count"] < 10000]

            # Only display the hidden gem if suitable book is found based on review numbers
            if not gem_pool.empty:
                hidden_gem = gem_pool.sort_values(by="avg_rating", ascending=False).iloc[0]
                if hidden_gem["work_id"] not in highlighted_ids:
                    highlighted_books.append(("Hidden Gem", hidden_gem))
                    highlighted_ids.append(hidden_gem["work_id"])

        # Display highlighted books
        if highlighted_books:
            for label, row in highlighted_books:
                st.markdown(f"###### {label}")
                unique_key = str(row['work_id']) + "_highlight"
                display_book(row, unique_key, ratings, review_summaries, reviews, books, highlighted=True)

        st.markdown("<hr style='border-top: 1px dashed #bbb;'>", unsafe_allow_html=True)
        
        st.markdown(f"###### More Recommendations")

        # Display remaining book recommendations
        remaining = filtered[~filtered['work_id'].isin(highlighted_ids)].reset_index(drop=True)
        count_to_show = min(st.session_state.show_count, len(remaining))
        st.write(f"Showing {count_to_show} of {len(remaining)}")

        for i in range(count_to_show):
            row = remaining.iloc[i]
            unique_key = str(row['work_id'])
            display_book(row, unique_key, ratings, review_summaries, reviews, books)

        # Option to click and show more books
        if st.session_state.show_count < len(remaining):
            if st.button("Show More"):
                st.session_state.show_count += 10
                st.rerun()

    # ------------------- RIGHT COLUMN: SUITCASE ---------------------
    with right_col:
        st.markdown("### Your Suitcase")

        if len(st.session_state.suitcase) == 0:
            st.write("Pack some books to get started!")
        else:
            for idx, book in enumerate(st.session_state.suitcase):
                title_col, button_col = st.columns([4, 1])
                
                with title_col:
                    st.markdown(f"**{book['Title']}** by *{book['Author']}*")

                with button_col:
                    # If x button is clicked, remove item from suitcase list
                    if st.button("X", key=f"remove_{idx}", use_container_width=True):
                        st.session_state.suitcase.pop(idx)
                        st.rerun()

        # Display suitcase summary if book has been added
        if len(st.session_state.suitcase) > 0:
            
            total_books = len(st.session_state.suitcase)

            total_hours = sum([
                b.get('est_reading_hours', 0)
                for b in st.session_state.suitcase
                if pd.notna(b.get('est_reading_hours'))
            ])

            st.markdown("---")
            st.markdown(f"**Total books packed:** {total_books}  \n**Estimated reading time:** {round(total_hours)} hrs")


            # ----------------- Export Suitcase -------------------
            suitcase_df = pd.DataFrame(st.session_state.suitcase)
            suitcase_df['Read?'] = ''  # Add a blank column to create checklist
            export_df = suitcase_df.drop(columns=['est_reading_hours', 'work_id'], errors='ignore')

            csv = export_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="🧳 Save Suitcase 🧳",
                data=csv,
                file_name="summer_shelf_suitcase.csv",
                mime='text/csv',
                use_container_width=True
            )
