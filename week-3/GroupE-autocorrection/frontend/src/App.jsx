import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [inputText, setInputText] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [selectedWord, setSelectedWord] = useState(null);
  const [lcsData, setLcsData] = useState(null);

  useEffect(() => {
    if (inputText) {
      axios.get(`http://127.0.0.1:8000/suggestions?input_word=${inputText}`)
        .then(response => setSuggestions(response.data));
    }
  }, [inputText]);

  const handleSuggestionClick = (word) => {
    setInputText(word.word);
    setSelectedWord(word.word);
    axios.get(`http://127.0.0.1:8000/lcs-table?input_word=${inputText}&chosen_word=${word.word}`)
      .then(response => setLcsData(response.data));
  };

  return (
    <div className="container">
      <h1>Auto Completion & Correction</h1>
      <input
        type="text"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Type a word..."
      />
      <div className="suggestions">
        <h2>Suggestions:</h2>
        {suggestions.map((s, index) => (
          <button key={index} onClick={() => handleSuggestionClick(s)}>
            {s.word} (Similarity: {s.similarity})
          </button>
        ))}
      </div>
      {lcsData && (
        <div className="lcs-table">
          <h2>LCS Table:</h2>
          <table>
            <tbody>
              {lcsData.table.map((row, i) => (
                <tr key={i}>
                  {row.map((cell, j) => (
                    <td key={j}>
                      {cell} {lcsData.arrows[i][j]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p>Similarity Score: {lcsData.similarity}</p>
        </div>
      )}
    </div>
  );
}

export default App;
