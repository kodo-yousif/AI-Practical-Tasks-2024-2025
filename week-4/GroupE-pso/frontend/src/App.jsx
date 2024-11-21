import { useState } from 'react';
import ParticleVisualization from './ParticleVisualization';
import './index.css';
import Form from './Form';
import axios from "axios";

export const startSimulation = async (params) => {
  const API_BASE_URL = "http://127.0.0.1:8000";
  try {
    const response = await axios.post(`${API_BASE_URL}/pso`, params);
    return response.data;
  } catch (error) {
    console.error("Error starting simulation:", error);
    throw error;
  }
};

function App() {
  const [simulationData, setSimulationData] = useState(null);

  const [params, setParams] = useState({
    num_particles: 10,
    goal_x: 0,
    goal_y: 0,
    cognitive_coeff: 1.0,
    social_coeff: 1.0,
    inertia: 1.2,
    iterations: 5,
  });

  // simple goal 
  // num_particles: 10,
  // goal_x: 5,
  // goal_y: 5,
  // cognitive_coeff: 1.5,
  // social_coeff: 1.5,
  // inertia: 0.5,
  // iterations: 50,

  // large particle swarm
  // num_particles: 50,
  // goal_x: 10,
  // goal_y: -10,
  // cognitive_coeff: 2.0,
  // social_coeff: 2.0,
  // inertia: 0.8,
  // iterations: 100,

  // high inertia weight
  // num_particles: 20,
  // goal_x: -5,
  // goal_y: -5,
  // cognitive_coeff: 1.0,
  // social_coeff: 1.0,
  // inertia: 1.2,
  // iterations: 50,

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setParams({ ...params, [name]: parseFloat(value) });
  };

  const handleStartSimulation = async () => {
    const API_BASE_URL = "http://127.0.0.1:8000";

    try {
      const response = await axios.post(`${API_BASE_URL}/pso`, params);
      setSimulationData(response?.data?.data)
    } catch (error) {
      console.error("Error starting simulation:", error);
      throw error;
    }
  };

  return (
    <div className="w-svw h-svh px-8 py-6 flex flex-row overflow-hidden">

      <div className='flex flex-col w-[50%]  gap-y-12'>
        <h1 className='text-2xl font-semibold'>Particle Swarm Optimization (PSO) Simulation</h1>
        <Form
          params={params}
          setParams={setParams}
          handleInputChange={handleInputChange}
          handleStartSimulation={handleStartSimulation}
        />
      </div>

      <div className='w-[50%]  flex flex-col gap-y-8'>
        <h1 className='text-2xl font-semibold'>Output</h1>
          {simulationData && <ParticleVisualization data={simulationData?.iterations} />}
      </div>
    </div>
  );
}

export default App;