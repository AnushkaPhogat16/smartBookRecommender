import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

import gradio as gr

load_dotenv()

#─── Load books and prepare thumbnails ────────────────────────────────────────
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

#─── Load and split the descriptions ───────────────────────────────────────────
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

#─── Build Chroma index with HuggingFace embeddings ───────────────────────────
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding=embedding)

#─── Recommendation logic ──────────────────────────────────────────────────────
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    # Unpack (Document, score) tuples
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [
        int(doc.page_content.strip('"').split()[0])
        for doc, score in recs
    ]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    recs = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recs.iterrows():
        # build caption
        desc_words = row["description"].split()
        truncated = " ".join(desc_words[:30]) + "..."
        authors = row["authors"].split(";")
        if len(authors) == 2:
            authors_str = f"{authors[0]}, and {authors[1]}"
        elif len(authors) > 2:
            authors_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
        else:
            authors_str = authors[0]
        caption = f"{row['title']} by {authors_str}: {truncated}"

        # Gallery wants [image_url, caption]
        results.append([row["large_thumbnail"], caption])
    return results

#─── Gradio UI ─────────────────────────────────────────────────────────────────
categories = ["All"] + sorted(books["simple_categories"].unique())
tones      = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Soft()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query       = gr.Textbox(label="What kind of book do you want to read?", placeholder="e.g., A story about magic!")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown     = gr.Dropdown(choices=tones,      label="Select an emotional tone:", value="All")
        submit_button     = gr.Button("Find recommendations!")

    gr.Markdown("## Here are your recommendations. Happy Reading <3")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()
