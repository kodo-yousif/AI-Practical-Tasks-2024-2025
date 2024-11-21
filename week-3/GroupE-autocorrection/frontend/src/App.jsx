<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
=======
>>>>>>> fdab320 (-a)
>>>>>>> bd1ef6d (-a)
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
<<<<<<< HEAD
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
=======
>>>>>>> fdab320 (-a)
>>>>>>> bd1ef6d (-a)

function App() {
  const [inputText, setInputText] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [selectedWord, setSelectedWord] = useState(null);
  const [lcsData, setLcsData] = useState(null);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
=======
>>>>>>> bd1ef6d (-a)
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
=======
  const [showTable, setShowTable] = useState(false)
>>>>>>> fdab320 (-a)

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
<<<<<<< HEAD
      )}
>>>>>>> ad973c8 (feat - setting up the basic frontend and backend stuff)
=======
<<<<<<< HEAD
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
=======
      </div>


      {showTable && lcsData ? <LscTable elementRef={ref} lcsData={lcsData} inputText={inputText} selectedWord={selectedWord} /> : ''}
>>>>>>> fdab320 (-a)
>>>>>>> bd1ef6d (-a)
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
<<<<<<< HEAD
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
=======
>>>>>>> fdab320 (-a)
>>>>>>> bd1ef6d (-a)
