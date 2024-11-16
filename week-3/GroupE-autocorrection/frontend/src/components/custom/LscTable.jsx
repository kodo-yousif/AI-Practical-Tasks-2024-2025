import BasicTable from "../ui/BasicTable"

function LscTable({ elementRef, lcsData, inputText, selectedWord }) {

    let i = inputText.split('')
    i = [' ', ...i]
    let j = selectedWord.split('')
    j = [' ', ...j]

    return (
        <section
            className='w-3/4 m-auto py-12'
            ref={elementRef}>
            <h1 className="text-2xl font-semibold border-b-2 w-max py-1">LCS Table</h1>
            {lcsData && (
                <div className="lcs-table flex flex-col gap-y-4">
                    <BasicTable data={lcsData} i={i} j={j} />
                    <span >Similarity Score: {lcsData.similarity}</span>

                </div>
            )}
        </section>
    )
}

export default LscTable