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
        similarity = lcs_similarity(input_word, word)
        suggestions.append({"word": word, "similarity": similarity})

    # Sort by highest similarity and return top 5
    suggestions = sorted(suggestions, key=lambda x: -x["similarity"])[:5]
    return suggestions


# end point for generating lsc table
@app.get("/lcs-table")
def get_lcs_table(input_word: str, chosen_word: str):
    
    # length of dimensions (words)
    m, n = len(input_word), len(chosen_word)
    
    # 2d table with dimension (m+1) * (n+1) each cell with a value of similarity between each corresponding character from each words
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 2d table with arrow indicating the direction of the solution path
    arrows = [[""] * (n + 1) for _ in range(m + 1)]


    # Fill the LCS table and arrows
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if input_word[i - 1] == chosen_word[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                arrows[i][j] = "↖"  # character match
            elif dp[i - 1][j] >= dp[i][j - 1]:
                dp[i][j] = dp[i - 1][j]
                arrows[i][j] = "↑"  # Up arrow
            else:
                dp[i][j] = dp[i][j - 1]
                arrows[i][j] = "←"  # Left arrow
                
                # we select between up and left by which one has the maximum value

    return {
        "table": dp,
        "arrows": arrows,
        "similarity": dp[m][n]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
