export type SudokuBoard = number[][];

export interface GAParams {
  board: SudokuBoard;
  population_size: number;
  generations: number;
  mutation_rate: number;
  max_no_improvement: number;
}

export interface GAResponse {
  generation: number;
  fitness: number;
  board: SudokuBoard;
  is_complete: boolean;
  message?: string;
}

export interface CombinedGAResponse {
  final_solution: GAResponse;
  history: GAResponse[];
}

export interface HistoryEntry extends GAResponse {
  timestamp?: string;
}