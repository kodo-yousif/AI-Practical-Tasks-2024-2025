import axios from 'axios';

const API_BASE_URL = "http://127.0.0.1:8000";  // URL of the FastAPI backend

export const startSimulation = async (params) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/start_simulation`, params);
    return response.data;
  } catch (error) {
    console.error("Error starting simulation:", error);
    throw error;
  }
};
