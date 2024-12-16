/* eslint-disable react/prop-types */import { useState, useEffect } from "react";
import { FaChessBishop } from "react-icons/fa";
import "./App.css";

const ChessBoard = ({ solution, boardSize, isPlaceholder, darkMode }) => {
    const [selectedBishop, setSelectedBishop] = useState(null);

    useEffect(() => {
        setSelectedBishop(null);
    }, [solution]);

    const handleBishopClick = (index) => {
        setSelectedBishop(index === selectedBishop ? null : index);
    };

    function handleOthersClick() {
        setSelectedBishop(null);
    }

    const isDiagonal = (index) => {
        if (selectedBishop === null) return false;
        const selectedRow = Math.floor(selectedBishop / boardSize);
        const selectedCol = selectedBishop % boardSize;
        const row = Math.floor(index / boardSize);
        const col = index % boardSize;
        if (selectedRow === row || selectedCol === col) return false;
        if (Math.abs(selectedRow - row) !== Math.abs(selectedCol - col)) return false;

        // Check if there is another bishop on the diagonal path before the current cell
        const rowStep = selectedRow < row ? 1 : -1;
        const colStep = selectedCol < col ? 1 : -1;
        let r = selectedRow + rowStep;
        let c = selectedCol + colStep;
        while (r !== row && c !== col) {
            if (solution.includes(r * boardSize + c)) return false;
            r += rowStep;
            c += colStep;
        }
        return true;
    };

    return (
        <div className={darkMode ? "chessboard-wrapper" : "chessboard-wrapper chessboard-wrapper-light"}>
            <h2 className={darkMode ? "chessboard-title" : "chessboard-title chessboard-title-light"}>Board</h2>
            <div className="bishop-positions">
                <span>Bishop Positions: [</span>
                {solution.map((index) => {
                    const row = boardSize - Math.floor(index / boardSize);
                    const col = index % boardSize;
                    const letter = String.fromCharCode(65 + col);
                    const position = `${letter}${row}`;
                    return <div key={index}>{position}</div>;
                })}
                <span>]</span>
            </div>

            {selectedBishop !== null && (
                <div className="selected-bishop-info">
                    <span className="selected-bishop-label">Selected Bishop: </span>
                    <div className="selected-bishop-position">
                        {String.fromCharCode(65 + (selectedBishop % boardSize))}
                        {boardSize - Math.floor(selectedBishop / boardSize)}
                    </div>
                </div>
            )}

            {isPlaceholder ? (
                <p>Click Generate Solutions to display the updated board!</p>
            ) : (
                <div className="chessboard-container">
                    <div className="chessboard-numbers">
                        {Array.from({ length: boardSize }).map((_, index) => (
                            <div key={index} className="chessboard-number">
                                {boardSize - index}
                            </div>
                        ))}
                    </div>
                    <div
                        className="chessboard"
                        style={{
                            gridTemplateColumns: `repeat(${boardSize}, 1fr)`,
                            gridTemplateRows: `repeat(${boardSize}, 1fr)`,
                        }}
                    >
                        {Array.from({ length: boardSize ** 2 }).map((_, index) => {
                            const row = Math.floor(index / boardSize);
                            const col = index % boardSize;
                            const isBishop = solution.includes(index);
                            const isEvenCell = (row + col) % 2 === 0;
                            const cellClass = isEvenCell ? "even-cell" : "odd-cell";
                            const isSelected = index === selectedBishop;
                            const isDiagonalCell = isDiagonal(index);
                            const isThreatenedCell = isDiagonalCell && isBishop;

                            return (
                                <div
                                    key={index}
                                    className={`cell ${cellClass} ${isBishop ? "bishop-cell" : ""} ${isSelected ? "selected-bishop" : ""} ${isDiagonalCell ? "diagonal-cell" : ""} ${isThreatenedCell ? "threatened-cell" : ""}`}
                                    onClick={() => isBishop ? handleBishopClick(index) : handleOthersClick()}
                                >
                                    <span>{isBishop && <FaChessBishop />}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
            <div className="chessboard-letters">
                <span className="spacer" />
                {Array.from({ length: boardSize }).map((_, index) => {
                    const letter = String.fromCharCode(65 + index);
                    return (
                        <div key={index} className="chessboard-letter">
                            {letter}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default ChessBoard;