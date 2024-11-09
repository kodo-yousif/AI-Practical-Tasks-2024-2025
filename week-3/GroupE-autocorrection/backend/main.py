from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import difflib

app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample word list (You can replace this with a larger list)
word_list = [
    "apple", "application", "apply", "ape", "apricot",
    "banana", "bandana", "grape", "pineapple", "peach"
]

def lcs_similarity(input_word: str, suggestion: str) -> int:
    """Calculate the Longest Common Subsequence (LCS) similarity."""
    m, n = len(input_word), len(suggestion)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if input_word[i - 1] == suggestion[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]

@app.get("/suggestions")
def get_suggestions(input_word: str) -> List[dict]:
    """Return top 5 word suggestions based on LCS similarity."""
    suggestions = []
    for word in word_list:
        similarity = lcs_similarity(input_word, word)
        suggestions.append({"word": word, "similarity": similarity})

    # Sort by highest similarity and return top 5
    suggestions = sorted(suggestions, key=lambda x: -x["similarity"])[:5]
    return suggestions

@app.get("/lcs-table")
def get_lcs_table(input_word: str, chosen_word: str):
    """Return LCS table data for visualization."""
    m, n = len(input_word), len(chosen_word)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    arrows = [[""] * (n + 1) for _ in range(m + 1)]

    # Fill the LCS table and arrows
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if input_word[i - 1] == chosen_word[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                arrows[i][j] = "↖"  # Diagonal arrow
            elif dp[i - 1][j] >= dp[i][j - 1]:
                dp[i][j] = dp[i - 1][j]
                arrows[i][j] = "↑"  # Up arrow
            else:
                dp[i][j] = dp[i][j - 1]
                arrows[i][j] = "←"  # Left arrow

    return {
        "table": dp,
        "arrows": arrows,
        "similarity": dp[m][n]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
