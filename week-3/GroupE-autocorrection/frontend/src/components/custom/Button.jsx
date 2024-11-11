import axios from "axios";
import { motion } from "framer-motion"
import toast from "react-hot-toast"

function Button({ tableRef, setShowTable, selectedWord, inputText, setLcsData  }) {

    const handleGenerateLcsTable = () => {
        axios.get(`http://127.0.0.1:8000/lcs-table?input_word=${inputText}&chosen_word=${selectedWord}`)
            .then(response => setLcsData(response.data));
    };
    return (
        <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className='hover:bg-white hover:text-black border-[2px] flex h-max self-end text-2xl py-2 px-4 rounded-xl  gap-x-4 justify-between items-center'
            onClick={() => {
                handleGenerateLcsTable();
                if(selectedWord){
                    setShowTable(true)
                    tableRef?.current?.scrollIntoView({ behavior: 'smooth' })
                }else
                    toast.error('Please select a Suggested word')
            }}
        >
            <span>
                Generate LCS Table
            </span>

            <svg
                fill="currentColor"
                viewBox="0 0 16 16"
                height="1em"
                width="1em"
            >
                <path d="M2 0a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2V2a2 2 0 00-2-2H2zm6.5 4.5v5.793l2.146-2.147a.5.5 0 01.708.708l-3 3a.5.5 0 01-.708 0l-3-3a.5.5 0 11.708-.708L7.5 10.293V4.5a.5.5 0 011 0z" />
            </svg>

        </motion.button>
    )
}

export default Button