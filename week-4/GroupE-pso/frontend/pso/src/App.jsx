import React, { useState } from 'react';
import ParticleVisualization from './ParticleVisualization';
import { startSimulation } from './api';
import './styles.css';

function App() {
  const [params, setParams] = useState({
    num_particles: 30,
    goal_x: 5,
    goal_y: 5,
    cognitive_coeff: 1.5,
    social_coeff: 1.5,
    inertia: 0.5,
    iterations: 100,
  });
  const [simulationData, setSimulationData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setParams({ ...params, [name]: parseFloat(value) });
  };

  const handleStartSimulation = async () => {
    setLoading(true);
    try {
      const data = await startSimulation(params);
      setSimulationData(data.results);
    } catch (error) {
      console.error("Simulation failed:", error);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Particle Swarm Optimization (PSO) Simulation</h1>
      <div className="controls">
        <label>
          Number of Particles:
          <input type="number" name="num_particles" value={params.num_particles} onChange={handleInputChange} />
        </label>
        <label>
          Goal X:
          <input type="number" name="goal_x" value={params.goal_x} onChange={handleInputChange} />
        </label>
        <label>
          Goal Y:
          <input type="number" name="goal_y" value={params.goal_y} onChange={handleInputChange} />
        </label>
        <label>
          Cognitive Coefficient:
          <input type="number" name="cognitive_coeff" value={params.cognitive_coeff} onChange={handleInputChange} />
        </label>
        <label>
          Social Coefficient:
          <input type="number" name="social_coeff" value={params.social_coeff} onChange={handleInputChange} />
        </label>
        <label>
          Inertia:
          <input type="number" name="inertia" value={params.inertia} onChange={handleInputChange} />
        </label>
        <label>
          Iterations:
          <input type="number" name="iterations" value={params.iterations} onChange={handleInputChange} />
        </label>
        <button onClick={handleStartSimulation} disabled={loading}>
          {loading ? "Running..." : "Start Simulation"}
        </button>
      </div>

      {simulationData && <ParticleVisualization data={simulationData} />}
    </div>
  );
}

export default App;
