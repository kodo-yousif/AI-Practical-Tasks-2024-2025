import axios from 'axios';
import { GAParams, CombinedGAResponse } from '../types/sudoku';

const API_BASE_URL = 'http://localhost:8000';

export const startGASolver = async (params: GAParams): Promise<CombinedGAResponse> => {
  const response = await axios.post(`${API_BASE_URL}/start_ga`, params);
  return response.data;
};