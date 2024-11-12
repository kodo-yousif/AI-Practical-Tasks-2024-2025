function Suggestions({ suggestions, setSelectedWord }) {

    return (
        <div className="flex flex-col gap-y-4 min-h-72 transition-all">
            <h2 className='font-bold'>Suggestions:</h2>
            <div className='flex flex-col gap-y-4 items-end '>
                {suggestions.map((s, index) => {
                    return (
                        <button
                            key={index}
                            onClick={() => {
                                setSelectedWord(s.word);
                            }}
                            className='outline font-semibold flex justify-between outline-white text-white  text-start w-80 hover:bg-white hover:outline-none hover:text-black px-4 py-1'>
                            <div>
                                {s.word}
                            </div>

                            <div>
                                Similarity: {s.similarity}
                            </div>
                        </button>
                    )
                })}
            </div>
        </div>
    )
}

export default Suggestions