<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
import { useState, useRef } from 'react';
import Suggestions from './components/custom/Suggestions';
import Button from './components/custom/Button';
import LscTable from './components/custom/LscTable';
import Home from './components/custom/Home';
<<<<<<< HEAD
=======
import React, { useState, useEffect } from 'react';
import axios from 'axios';
>>>>>>> ad973c8 (feat - setting up the basic frontend and backend stuff)
=======
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))

function App() {
  const [inputText, setInputText] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [selectedWord, setSelectedWord] = useState(null);
  const [lcsData, setLcsData] = useState(null);
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
  const [showTable, setShowTable] = useState(false)

  const ref = useRef()

  return (
    <div className='overflow-x-hidden'>
      <div className="h-svh w-svw flex flex-col justify-center gap-y-16 font-mono">

        <h1 className='text-4xl text-center'>Auto Completion & Correction</h1>
        <Home
          inputText={inputText}
          setInputText={setInputText}
          setSuggestions={setSuggestions}
          setLcsData={setLcsData}
          selectedWord={selectedWord}
          setSelectedWord={setSelectedWord}
          setShowTable={setShowTable}
        />


        <div className='flex justify-between w-3/4 mx-auto'>
          <Suggestions
            suggestions={suggestions}
            setSelectedWord={setSelectedWord}
            inputText={inputText}
            setLcsData={setLcsData}
          />

          {suggestions.length > 0 && <Button tableRef={ref} setShowTable={setShowTable} selectedWord={selectedWord} inputText={inputText} setLcsData={setLcsData} />}
        </div>
      </div>


      {showTable && lcsData ? <LscTable elementRef={ref} lcsData={lcsData} inputText={inputText} selectedWord={selectedWord} /> : ''}
<<<<<<< HEAD
=======

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
>>>>>>> ad973c8 (feat - setting up the basic frontend and backend stuff)
=======
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
    </div>
  );
}

<<<<<<< HEAD
<<<<<<< HEAD
export default App;
=======
export default App;
>>>>>>> ad973c8 (feat - setting up the basic frontend and backend stuff)
=======
export default App;
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
