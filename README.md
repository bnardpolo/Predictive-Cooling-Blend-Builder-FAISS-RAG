Predictive Cooling Blend Builder with FAISS-RAG

An intelligent flavor compound recommendation system that combines machine learning, semantic search, and heuristic filtering to design optimal cooling flavor blends.

Overview

The Predictive Cooling Blend Builder is a data-driven application that helps flavor scientists and product developers discover and combine cooling compounds for food, beverage, and personal care. By analyzing more than 14,000 aroma and taste molecules, the system predicts sensory profiles, filters by key attributes such as coolness intensity and bitterness, and suggests compatible compound combinations.

Key Capabilities

Intelligent compound discovery with natural-language search across 14,000+ molecules

Multi-factor scoring that balances coolness intensity, bitterness control, and keyword relevance

Context-aware recommendations for dessert, herbal, or mint flavor profiles

Machine-learning predictions using a Random Forest classifier for taste attributes

Interactive blend building with options to assemble and export custom formulations

Flexible data access from local files or AWS S3


┌─────────────────────────────────────────────────────┐
│                 Streamlit UI Layer                  │
│                 (ui/stream_app.py)                  │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│                 Core Pipeline Logic                 │
│              (src/flavor_rag_main.py)              │
│  • Scope inference (dessert/herbal/mint)           │
│  • Threshold-based filtering                        │
│  • Multi-factor scoring engine                      │
│  • S3/local file management                         │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│                   Data & ML Assets                  │
│  • Aroma database (5,046 molecules)                 │
│  • Taste database (8,982 molecules)                 │
│  • Random Forest taste classifier                   │
│  • FAISS vector index (ready for integration)       │
└─────────────────────────────────────────────────────┘
How It Works
1. Scope inference

The system infers intent from the goal text:

"ice cream with vanilla" → dessert profile
"herbal camphor cooling" → herbal mint profile
"maximum menthol sensation" → pure cooling profile
2. Multi-factor scoring

Each compound receives a composite score:

| Factor             | Weight | Description                                 |
| ------------------ | :----: | ------------------------------------------- |
| Coolness intensity |   0.6  | Cooling sensation on a 0–1 scale            |
| Bitterness penalty |  -0.5  | Inverse of predicted bitterness probability |
| Keyword match      |   0.6  | Alignment with the user goal                |
| Scope bonus        |   0.3  | Relevance to the inferred profile           |



3. Adaptive filtering

Initial thresholds:
  bitter_threshold ≤ 0.40
  cool_score ≥ 0.15

If no strong matches are found, the system relaxes constraints and returns the nearest alternatives.
Dataset Overview
Compound databases

| File                     | Molecules | Key attributes                     |
| ------------------------ | :-------: | ---------------------------------- |
| Aroma_molecules_5046.csv |   5,046   | Aromatic profiles, volatility      |
| Taste_molecules_8982.csv |   8,982   | Taste classes, sensory descriptors |
| *_clean.csv              |  Combined | Normalized, ML-ready features      |

Example compound entry

{
  "name": "Menthyl Lactate",
  "aroma": ["mint", "cool", "fresh"],
  "taste": ["cooling", "slightly sweet"],
  "cool_score": 0.85,
  "bitter_prob": 0.12,
  "predicted_classes": ["mint", "fresh", "cooling"]
}


User Interface
Main panel

Design goal input using natural language

Intent presets (Max Cooling, Herbal Mint, Dessert, Custom)

Results displayed as cards with key metrics and reasoning

Sidebar controls

Bitter threshold

Top-K retrieval

Minimum cool score

Preset compounds seeded from safe defaults

Blend manager

Add compounds to a working blend

Export blend compound names as a .txt file

Visual badges for quick review of cool and bitter scores

Example Usage
Dessert-safe cooling


Goal: "vanilla ice cream with mild cooling, no harsh bitterness"

Detected scope: dessert
Found 8 candidates (cool ≥ 0.15, bitter ≤ 0.40)

Top suggestions:
1. Menthyl Lactate (cool=0.85, bitter=0.12)
2. Ethyl Menthane Carboxamide (cool=0.78, bitter=0.18)
3. Menthone Glycerol Acetal (cool=0.65, bitter=0.22)
Herbal mint blend
Goal: "eucalyptus and peppermint with strong camphor notes"

Detected scope: herbal mint
Prioritized keywords: camphor, eucalyptus, menthol

Top suggestions:
1. Eucalyptol (cool=0.72, bitter=0.08)
2. Menthol (cool=0.95, bitter=0.28)
3. Camphor (cool=0.68, bitter=0.35)
Machine Learning Integration
Current state

Random Forest taste classifier trained on 8,982 molecules

Multi-label binarizer for taste class predictions

Feature engineering pipeline established

Roadmap

Integrate taste-model predictions into scoring

Add FAISS semantic search for query understanding

Hybrid scoring: ML predictions + heuristics + vector similarity

Compound-compatibility predictor for synergy and clash detection

# Planned: hybrid recommendation pipeline
def answer_hybrid(goal_text):
    candidates = faiss_search(goal_text, k=50)            # semantic search
    enhanced = ml_predict_taste(candidates)               # ML predictions
    shortlist = apply_thresholds(enhanced)                # heuristic filter
    return rank_by_multi_factor_score(shortlist)          # composite rank
Project Structure

Predictive-Cooling-Blend-Builder-FAISS-RAG/
├── src/
│   ├── flavor_rag_main.py
│   └── genai_flavor_rag.py
├── ui/
│   └── stream_app.py
├── models/
│   ├── taste_model_randomforest.joblib
│   └── taste_label_binarizer.joblib
├── faiss_index/
├── data/
│   ├── Aroma_molecules_5046.csv
│   ├── Taste_molecules_8982.csv
│   ├── aroma_clean.csv
│   ├── taste_clean.csv
│   └── taste_metrics.json
├─
├── requirements.txt
├── .env
├── Dockerfile
└── README.md



| Layer         | Technology                        |
| ------------- | --------------------------------- |
| Frontend      | Streamlit                         |
| ML framework  | scikit-learn (Random Forest)      |
| Vector search | FAISS                             |
| Data          | pandas, numpy                     |
| Cloud         | AWS S3 with boto3                 |
| Deployment    | Docker, Streamlit Community Cloud |



Advanced Features
S3 integration

Works with both local and S3 paths. Artifacts are downloaded once and cached locally for subsequent runs.
FLAVOR_DATASET = "s3://my-bucket/data/taste_clean.csv"


Scope-based keyword boosting
SCOPE_KEYWORDS = {
    "dessert": ["vanilla", "creamy", "sweet", "buttery"],
    "herbal":  ["camphor", "eucalyptus", "sage"],
    "mint":    ["menthol", "peppermint", "cool"]
}

Preset compounds

Pre-vetted options emphasizing higher cooling (≥ 0.6), lower bitterness (≤ 0.3), and common industry use.

Contributing

Contributions are welcome. Priority areas include:

Integrating taste-model predictions into scoring

Implementing FAISS-based semantic search

Improving UI visualizations for sensory profiles

Expanding the compound databases

Adding unit tests for core pipeline components

License

Add your license here (for example, MIT or Apache-2.0).

Acknowledgments

Compound information aggregated from public flavor chemistry research

Inspired by Retrieval-Augmented Generation architectures

Built with Streamlit components

Contact

Developer: bnardpolo
GitHub: Predictive-Cooling-Blend-Builder-FAISS-RAG
Issues: Report bugs or request features

