import { useState } from 'react'
import './App.css'

function App() {
  const [solutions, setSolutions] = useState([])
  const [loading, setLoading] = useState(false)
  const [selectedSolution, setSelectedSolution] = useState(null)
  const [error, setError] = useState(null)

  const generateMagicSquare = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('http://localhost:5000/generate')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setSolutions(data.solutions)
      if (data.solutions.length > 0) {
        setSelectedSolution(data.solutions[0].individual)
      }
    } catch (error) {
      console.error('Error:', error)
      setError('Failed to generate magic square. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const renderGrid = () => {
    if (!selectedSolution) return (
      <div className="grid" style={{ opacity: 0.5 }}>
        {[0, 1, 2].map(row => (
          <div key={row} className="row">
            {[0, 1, 2].map(col => (
              <div key={col} className="cell">
                ?
              </div>
            ))}
          </div>
        ))}
      </div>
    )

    return (
      <div className="grid">
        {[0, 1, 2].map(row => (
          <div key={row} className="row">
            {[0, 1, 2].map(col => (
              <div key={col} className="cell">
                {selectedSolution[row * 3 + col]}
              </div>
            ))}
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="container">
      <h1>Magic Square Generator</h1>
      <div className="content">
        <div className="controls">
          <button 
            onClick={generateMagicSquare} 
            disabled={loading}
          >
            {loading ? 'Generating Magic Square...' : 'Generate Magic Square'}
          </button>
          
          <div className="solutions-list">
            {solutions.length === 0 ? (
              <div className="solution-item" style={{ opacity: 0.5 }}>
                No solutions generated yet. Click the button above to start!
              </div>
            ) : (
              solutions.map((solution, index) => (
                <div 
                  key={index} 
                  className="solution-item"
                  onClick={() => setSelectedSolution(solution.individual)}
                >
                  Generation {solution.generation}, Fitness: {solution.fitness}
                </div>
              ))
            )}
          </div>
        </div>
        
        <div className="grid-container">
          {renderGrid()}
          {error && <div className="error-message">{error}</div>}
        </div>
      </div>
    </div>
  )
}

export default App
