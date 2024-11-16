import axios from "axios";
import { useEffect } from "react";

function Home({ inputText, setInputText, setSuggestions, setLcsData, selectedWord, setSelectedWord }) {

    useEffect(() => {
        if (inputText) {
            axios.get(`http://127.0.0.1:8000/suggestions?input_word=${inputText}`)
                .then(response => setSuggestions(response.data));
        }
    }, [inputText]);

    return (
        <div className='w-[80%] mx-auto flex gap-x-2 textInput'>
            <input
                type="text"
                value={inputText}
                placeholder="Type a word..."
                className={` h-12 text-lg text-black rounded-md px-4  focus:outline-black focus:outline-2 focus:border-white `}
                onKeyDown={(e) => {
                    if (
                        e.code === "Minus" ||
                        ["+", "{", "}", ".", "$", "#", "@", "!", "~", "`", "'", '""', "%", "^", "&", "*", "(", ")", "=", "<", ">", "?", "/", ",", "|"].includes(e.key)
                    )
                        e.preventDefault();
                }}
                onChange={(e) => {
                    setInputText(e.target.value);
                    if (!e.target.value) {
                        setSelectedWord(null)
                        setSuggestions([]);
                        setLcsData(null)
                    }
                }
                }
            />

            <span className={`  ${!selectedWord ? 'bg-gray-600 opacity-90' : 'bg-blue-500'} transition-all flex items-center font-semibold justify-center w-48 h-12 rounded-md px-4 text-xl`}>
                <i>{selectedWord}</i>
            </span>
        </div>
    )
}

export default Home