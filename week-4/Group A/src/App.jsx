import { BsFillMoonStarsFill } from "react-icons/bs";
import { BsFillSunFill } from "react-icons/bs";
import { useEffect, useState } from "react";
import axios from "axios";
import ChessBoard from "./ChessBoard";
import SolutionList from "./SolutionList";
import "./App.css";
import { VscLoading } from "react-icons/vsc";

function App() {
  const [boardSize, setBoardSize] = useState(8);
  const [generation, setGeneration] = useState(100); // Add state for generation
  const [population, setPopulation] = useState(100); // Add state for population size
  const [mutationRate, setMutationRate] = useState(0.1); // Add state for mutation rate
  const [activeBoardSize, setActiveBoardSize] = useState(8);
  const [solutions, setSolutions] = useState([]);
  const [selectedSolution, setSelectedSolution] = useState(null);
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    const savedMode = localStorage.getItem("darkMode");
    return savedMode === null ? true : JSON.parse(savedMode);
  });

  useEffect(() => {
    localStorage.setItem("darkMode", JSON.stringify(darkMode));
  }, [darkMode]);

  const fetchSolutions = async () => {
    setLoading(true);
    setSolutions([]);
    setSelectedSolution(null);

    try {
      const response = await axios.post("http://127.0.0.1:8000/solutions", {
        board_size: boardSize,
        generations: generation,
        population_size: population,
        mutation_rate: mutationRate,
      });

      const { solutions: fetchedSolutions } = response.data;

      if (fetchedSolutions.length > 0) {
        const bestSolution = fetchedSolutions.reduce((prev, current) =>
          prev.fitness > current.fitness ? prev : current
        );
        setSelectedSolution(bestSolution.solution);
      }
      setSolutions(fetchedSolutions);
      setActiveBoardSize(boardSize);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleBoardSizeChange = (e) => {
    const size = Number(e.target.value);
    if (size > 0) {
      setBoardSize(size);
      document.documentElement.style.setProperty("--board-size", size);
    }
  };

  const handleGenerationChange = (e) => setGeneration(Number(e.target.value));
  const handlePopulationChange = (e) => setPopulation(Number(e.target.value));
  const handleMutationRateChange = (e) =>
    setMutationRate(Number(e.target.value));

  return (
    <div className={`app ${darkMode ? "dark-mode" : "light-mode"}`}>
      <header>
        <h1 className="title">N-Bishop Puzzle Solver</h1>
      </header>

      <div className="toggle-container" onClick={() => setDarkMode(!darkMode)}>
        {darkMode ? (
          <BsFillSunFill className="icon" />
        ) : (
          <BsFillMoonStarsFill className="icon" />
        )}
      </div>

      <div className="input-container">
        <div className="input-container">
          <label htmlFor="board-size-input">Board Size:</label>
          <input
            className="border-size"
            id="board-size-input"
            type="number"
            value={boardSize}
            onChange={handleBoardSizeChange}
            min="3"
          />
        </div>

        <div className="input-container">
          <label htmlFor="generation ">Generation Count:</label>
          <input
            className="border-size"
            id="generation"
            type="number"
            value={generation}
            onChange={handleGenerationChange}
            min="1"
          />
        </div>

        <div className="input-container">
          <label htmlFor="population">Population Size:</label>
          <input
            className="border-size"
            id="population"
            type="number"
            value={population}
            onChange={handlePopulationChange}
            min="1"
          />
        </div>
        <div className="input-container">
          <label htmlFor="mutationRate">Mutation Rate:</label>
          <input
            className="border-size"
            id="mutationRate"
            type="number"
            step="0.01"
            value={mutationRate}
            onChange={handleMutationRateChange}
            min="0"
            max="1"
          />
        </div>
      </div>

      <button onClick={fetchSolutions} className="fetch-button">
        Generate Solutions
      </button>

      {loading ? (
        <p className="loading-text">
          Loading solutions... <VscLoading />
        </p>
      ) : (
        <div className="container">
          <SolutionList
            solutions={solutions}
            onSelect={setSelectedSolution}
            defaultSolution={selectedSolution}
          />
          {selectedSolution && (
            <div className="chessboard-container">
              <ChessBoard
                solution={selectedSolution}
                boardSize={activeBoardSize}
                darkMode={darkMode}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
