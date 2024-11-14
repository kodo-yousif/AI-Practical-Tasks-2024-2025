<<<<<<< HEAD
#Framework for building APIs quickly and easily in Python.
from fastapi import FastAPI 

#Allows Cross-Origin Resource Sharing, enabling the backend to communicate with your React frontend.
from fastapi.middleware.cors import CORSMiddleware 

from typing import List
from words import word_list

app = FastAPI()

# Allow CORS for React frontend throw middleware(API)

#Why Use CORS Middleware?
# Without this middleware, your React frontend would face CORS errors when trying to fetch data from the FastAPI backend,
# due to the browser’s same-origin policy.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # allow requests from any domain
    allow_credentials=True, # allow cookies and authentication header in requests
    allow_methods=["*"], # allow http method
    allow_headers=["*"], # allow http header
)


def lcs_similarity(input_word: str, suggestion: str) -> int:
    
    m, n = len(input_word), len(suggestion)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # dp has dimensions (m+1) * (n+1) m,n are the length of input and suggestion
    # Each cell dp[i][j] stores the length of the LCS of the substrings input_word[0:i] and suggestion[0:j].
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if input_word[i - 1] == suggestion[j - 1]:
                # is characters match add the cells value by 1
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # if the characters doesn't match take the maximum value from either top[i-1] or left[j-1]
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # we are returning the length of LCS that is a similarity score between two words.
    # so with these method we are returning the similarity value between each two words that we are sending
    return dp[m][n]


#end point for suggestions
@app.get("/suggestions")
def get_suggestions(input_word: str) -> List[dict]:

    suggestions = []
    for word in word_list:
        # we are sending the input from user and the list of words in our words.py file
=======
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
>>>>>>> ad973c8 (feat - setting up the basic frontend and backend stuff)
        similarity = lcs_similarity(input_word, word)
        suggestions.append({"word": word, "similarity": similarity})

    # Sort by highest similarity and return top 5
    suggestions = sorted(suggestions, key=lambda x: -x["similarity"])[:5]
    return suggestions

<<<<<<< HEAD

# end point for generating lsc table
@app.get("/lcs-table")
def get_lcs_table(input_word: str, chosen_word: str):
    
    # length of dimensions (words)
    m, n = len(input_word), len(chosen_word)
    
    # 2d table with dimension (m+1) * (n+1) each cell with a value of similarity between each corresponding character from each words
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 2d table with arrow indicating the direction of the solution path
    arrows = [[""] * (n + 1) for _ in range(m + 1)]


=======
@app.get("/lcs-table")
def get_lcs_table(input_word: str, chosen_word: str):
    """Return LCS table data for visualization."""
    m, n = len(input_word), len(chosen_word)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    arrows = [[""] * (n + 1) for _ in range(m + 1)]

>>>>>>> ad973c8 (feat - setting up the basic frontend and backend stuff)
    # Fill the LCS table and arrows
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if input_word[i - 1] == chosen_word[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
<<<<<<< HEAD
                arrows[i][j] = "↖"  # character match
=======
                arrows[i][j] = "↖"  # Diagonal arrow
>>>>>>> ad973c8 (feat - setting up the basic frontend and backend stuff)
            elif dp[i - 1][j] >= dp[i][j - 1]:
                dp[i][j] = dp[i - 1][j]
                arrows[i][j] = "↑"  # Up arrow
            else:
                dp[i][j] = dp[i][j - 1]
                arrows[i][j] = "←"  # Left arrow
<<<<<<< HEAD
                
                # we select between up and left by which one has the maximum value
=======
>>>>>>> ad973c8 (feat - setting up the basic frontend and backend stuff)

    return {
        "table": dp,
        "arrows": arrows,
        "similarity": dp[m][n]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
