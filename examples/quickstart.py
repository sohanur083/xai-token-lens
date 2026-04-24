from token_lens import TokenLens

# Uses built-in proxy sentiment model (no heavy deps needed for demo).
# For real transformers, install [full] extras and use the HF model path.
lens = TokenLens("distilbert-base-uncased-finetuned-sst-2-english")

vis = lens.explain(
    "The movie was surprisingly boring and way too long, but the acting was great.",
    method="lime",
)
print(vis.summary())
print()
vis.show()
vis.to_html("explanation.html")
print("\nHTML report -> explanation.html")
